import json
from pathlib import Path
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from lip.data.index import read_frame
from lip.geometry.so3 import center_pose


class ClipDataset(Dataset):
    def __init__(self, root, index_root, length=8, split='train', seed=42, steps=1_000_000,
                 rank=0, world=1, batch=1, start=0, augmentation=True, fixed=None, synthetic_erasure=False):
        self.root=root;self.index=Path(index_root);self.length=length;self.seed=seed;self.steps=steps
        self.rank=rank;self.world=world;self.batch=batch;self.start=start;self.augmentation=augmentation
        self.synthetic_erasure=synthetic_erasure and augmentation and split=='train'
        self.audit=json.loads((self.index/'audit.json').read_text())
        self.streams=[json.loads(x) for x in (self.index/'streams.jsonl').read_text().splitlines() if json.loads(x)['split']==split]
        self.fixed=fixed;self.poses=[];self.meshes=[];mesh_cache={}
        for s in self.streams:
            with np.load(self.index/s['pose_cache']) as p:self.poses.append({k:p[k].copy() for k in p.files})
            if s['mesh_cache'] not in mesh_cache:
                with np.load(self.index/s['mesh_cache']) as p:mesh_cache[s['mesh_cache']]={k:p[k].copy() for k in p.files}
            self.meshes.append(mesh_cache[s['mesh_cache']])
        pool_path=self.index/f'{split}_clips_L{length}.json'
        if not pool_path.exists():raise RuntimeError(f'Missing sampling statistics: run tools/build_sampling.py --length {length}')
        self.pool=json.loads(pool_path.read_text())
        self.buckets={}
        for kind in ['moving','occluded','uniform']:
            groups={}
            for item in self.pool:
                if kind!='uniform' and not item[kind]:continue
                s=self.streams[item['stream']]
                groups.setdefault(str(s['object_id']),{}).setdefault(s['subject_id']+'/'+s['sequence_id'],{}).setdefault(s['camera_serial'],[]).append(item)
            self.buckets[kind]=groups
        self.available=[k for k in ['moving','occluded','uniform'] if self.buckets[k]]
        if not self.available:raise RuntimeError('No contiguous clips available')
        weights=dict(moving=.5,occluded=.25,uniform=.25)
        self.weights=np.array([weights[k] for k in self.available]);self.weights/=self.weights.sum()
        print(json.dumps(dict(sampler_available_buckets=self.available,weights=self.weights.tolist(),rank=rank)),flush=True)

    def __len__(self):return self.steps

    def choose(self, idx):
        # Index includes optimizer/microbatch stream position; resume reconstructs it.
        global_sample=(self.start+idx//self.batch)*self.batch*self.world+self.rank*self.batch+idx%self.batch
        rng=np.random.default_rng(self.seed+global_sample)
        if self.fixed is not None:return self.fixed[idx%len(self.fixed)],rng
        groups=self.buckets[str(rng.choice(self.available,p=self.weights))]
        for _ in range(3):groups=groups[str(rng.choice(list(groups)))]
        # Exact stride weights after validity filtering, with explicit finite pool.
        available=sorted(set(x['stride'] for x in groups));w=np.array([{1:.6,2:.3,4:.1}[s] for s in available]);w/=w.sum()
        stride=int(rng.choice(available,p=w));items=[x for x in groups if x['stride']==stride]
        return items[int(rng.integers(len(items)))],rng

    def __getitem__(self, idx):
        item,rng=self.choose(idx);sid=item['stream'];s=self.streams[sid];p=self.poses[sid];mesh=self.meshes[sid]
        # One extra initial pose before observation history and three rollout futures.
        frames=np.arange(item['start']-item['stride'],item['start']+(self.length+3)*item['stride'],item['stride'])
        loc=np.searchsorted(p['frames'],frames)
        if np.any(loc>=len(p['frames'])) or not np.array_equal(p['frames'][loc],frames):raise RuntimeError('Clip crosses a missing frame')
        poses=torch.from_numpy(p['poses'][loc])
        poses=center_pose(poses,torch.from_numpy(mesh['center']))
        images=[read_frame(self.root,s,int(f),self.audit['depth_scale_to_m']) for f in frames[1:]]
        rgb=torch.from_numpy(np.stack([x[0] for x in images]));depth=torch.from_numpy(np.stack([x[1] for x in images]))
        if self.augmentation:
            rgb=(rgb*float(rng.uniform(.9,1.1))+float(rng.uniform(-.025,.025))).clamp(0,1)
            if rng.random()<.2:
                rgb=torch.nn.functional.avg_pool2d(rgb,(1,3),stride=1,padding=(0,1))
            noise=torch.from_numpy(rng.normal(0,.0005,depth.shape).astype('f4'))
            depth=torch.where(depth>0,(depth+noise).clamp_min(0),depth)
            depth[torch.from_numpy(rng.random(depth.shape)<.01)]=0
        if self.synthetic_erasure and rng.random()<.2:
            # Segmentation only selects an augmentation location; it is never fed
            # to the model or used as the crop or the erasure's silhouette boundary.
            label=Path(self.root)/s['relative_dir']/f'labels_{int(frames[self.length]):06d}.npz'
            with np.load(label) as z:yy,xx=np.where(z['seg']==s['object_id'])
            if len(xx):
                q=int(rng.integers(len(xx)));h,w=depth.shape[-2:]
                coarse=rng.random((12,16)).astype('f4')
                irregular=cv2.resize(coarse,(w,h),interpolation=cv2.INTER_LINEAR)>.5
                y,x=np.ogrid[:h,:w];radius=float(rng.uniform(20,70))
                mask=torch.from_numpy(irregular & ((x-xx[q])**2+(y-yy[q])**2<radius**2))
                rgb[:,:,mask]=0.;depth[:,:,mask]=0.
        effective=self.length
        if self.augmentation and rng.random()<.2:effective=int(rng.choice(sorted(set([1,min(2,self.length),min(4,self.length)]))))
        return dict(rgb=rgb,depth=depth,poses=poses,frames=torch.from_numpy(frames),
                    times=torch.from_numpy(frames[1:].astype('f4')/self.audit['fps']),k=torch.tensor(s['intrinsics']),
                    mesh=mesh,stream=s,effective=effective,seed=int(rng.integers(2**31)),sample=item)


def collate(batch):return batch
