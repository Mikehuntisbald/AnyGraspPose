"""Continuous train fragments with an explicit initialization frame and no cross-stream memory."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from lip.geometry.so3 import center_pose


class StreamClips(Dataset):
    def __init__(self,root,index_root,burn_in=8,unroll=16,split='train',seed=42,length=1000000,
                 start_sample=0,rank=0,world=1,fixed=None,decode_threads=4,
                 external_initializers=None,external_initializers_sha256=None,real_initialization_probability=0.,
                 include_initial_observation=False):
        self.root=Path(root);self.index=Path(index_root);self.total=burn_in+unroll;self.seed=seed
        self.length=length;self.start_sample=start_sample;self.rank=rank;self.world=world
        self.decode_threads=decode_threads;self._pool=None
        self.audit=json.loads((self.index/'audit.json').read_text())
        self.streams=sorted([s for s in map(json.loads,(self.index/'streams.jsonl').read_text().splitlines()) if s['split']==split],key=lambda s:s['stream_id'])
        self.poses=[];self.starts=[];self.meshes={};self.groups={}
        for i,s in enumerate(self.streams):
            with np.load(self.index/s['pose_cache']) as z:p={k:z[k].copy() for k in z.files}
            n=len(p['frames']);starts=np.arange(max(0,n-self.total))
            # Every observation is contiguous; no hidden stride / missing-frame crossing.
            starts=np.array([j for j in starts if np.array_equal(p['frames'][j:j+self.total+1],np.arange(p['frames'][j],p['frames'][j]+self.total+1))],dtype='i8')
            self.poses.append(p);self.starts.append(starts)
            if len(starts):self.groups.setdefault(s['object_id'],{}).setdefault(s['subject_id']+'/'+s['sequence_id'],[]).append(i)
            if s['mesh_cache'] not in self.meshes:
                with np.load(self.index/s['mesh_cache']) as z:self.meshes[s['mesh_cache']]={k:z[k].copy() for k in z.files}
        self.fixed=fixed
        if not self.groups:raise RuntimeError('No continuous fragments in requested official split')
        self.real_initialization_probability=float(real_initialization_probability)
        if not 0<=self.real_initialization_probability<=1:raise ValueError('Invalid real initialization probability')
        self.include_initial_observation=bool(include_initial_observation);self.external=None
        if external_initializers:
            from lip.data.external_initializers import load_train_initializers
            if split!='train':raise ValueError('External training initializers cannot load into val/test clips')
            self.external=load_train_initializers(external_initializers,external_initializers_sha256,self.streams,self.audit)['initializers']
        if self.real_initialization_probability and (self.external is None or not self.include_initial_observation):
            raise ValueError('Real training initialization requires bound predictions and initial observation')

    def __len__(self):return self.length

    def __getstate__(self):
        result=self.__dict__.copy();result['_pool']=None;return result

    def choose(self,index):
        sample=self.start_sample+index*self.world+self.rank
        if self.fixed is not None:return self.fixed[sample%len(self.fixed)]
        rng=np.random.default_rng(self.seed+sample)
        obj=int(rng.choice(sorted(self.groups)));group=self.groups[obj]
        physical=sorted(group)[int(rng.integers(len(group)))];streams=group[physical]
        stream=streams[int(rng.integers(len(streams)))];starts=self.starts[stream]
        return dict(stream=stream,start=int(starts[int(rng.integers(len(starts)))]),seed=int(rng.integers(2**31)))

    def fixed_manifest(self,count=16):
        result=[];seen=set();i=0
        while len(result)<count:
            item=self.choose(i);key=(item['stream'],item['start']);i+=1
            if key in seen:continue
            seen.add(key);result.append(item)
        return result

    def __getitem__(self,index):
        item=self.choose(index);sid=item['stream'];s=self.streams[sid];p=self.poses[sid];start=item['start']
        loc=slice(start,start+self.total+1);frames=p['frames'][loc]
        if len(frames)!=self.total+1:raise ValueError('Fixed manifest has insufficient frames')
        # Honor captured timestamps when supplied; released DexYCB indexes have only FPS.
        times=p['timestamps'][loc].astype('f8') if 'timestamps' in p else frames.astype('f8')/self.audit['fps']
        if not np.all(np.diff(times)>0):raise ValueError('Nonmonotonic dataset timestamps')
        mesh=self.meshes[s['mesh_cache']]
        poses=center_pose(torch.from_numpy(p['poses'][loc].copy()),torch.from_numpy(mesh['center']))
        directory=self.root/s['relative_dir']
        def decode(frame):
            rgb=cv2.imread(str(directory/f'color_{int(frame):06d}.jpg'))
            depth=cv2.imread(str(directory/f'aligned_depth_to_color_{int(frame):06d}.png'),-1)
            if rgb is None or depth is None:raise RuntimeError('Missing RGB-D frame')
            return cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB).transpose(2,0,1),depth[None]
        if self._pool is None and self.decode_threads>1:
            cv2.setNumThreads(1);self._pool=ThreadPoolExecutor(self.decode_threads)
        requested_frames=frames if self.include_initial_observation else frames[1:]
        images=list(self._pool.map(decode,requested_frames)) if self._pool else [decode(f) for f in requested_frames]
        initial_image=images[0] if self.include_initial_observation else None
        if self.include_initial_observation:images=images[1:]
        result=dict(rgb=torch.from_numpy(np.stack([v[0] for v in images])),depth=torch.from_numpy(np.stack([v[1] for v in images])),
            initial_pose=poses[0].clone(),targets=poses[1:],timestamps=torch.from_numpy(times),frames=torch.from_numpy(frames),
            mesh=mesh,k=torch.tensor(s['intrinsics']),stream=s,sample=item,
            timestamp_source='captured timestamps' if 'timestamps' in p else 'frame_index / official FPS; release has no measured frame clock',
            depth_scale=float(self.audit['depth_scale_to_m']),nominal_frame_interval=1/float(self.audit['fps']))
        if initial_image is not None:
            result.update(initial_rgb=torch.from_numpy(initial_image[0]),initial_depth=torch.from_numpy(initial_image[1]))
        if self.external is not None:
            from lip.data.external_initializers import request_real_initializer,initializer_key
            requested=request_real_initializer(item['seed'],self.real_initialization_probability)
            result.update(real_initialization_requested=requested,real_initialization_missing=False)
            if requested:
                key=initializer_key(s['stream_id'],int(frames[0]))
                if key not in self.external:raise ValueError('Missing requested initializer record: '+key)
                entry=self.external[key];result['real_initialization_missing']=entry is None
                if entry is not None:
                    result['real_initial_pose_original']=torch.tensor(entry['pose_original'],dtype=torch.float32)
                    result['real_initial_pose']=center_pose(result['real_initial_pose_original'],torch.from_numpy(mesh['center']).float())
                    result['real_initializer_key']=key
        return result


def collate(samples):return samples
