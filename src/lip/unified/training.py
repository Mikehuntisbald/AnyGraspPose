"""Native episodes, real-textured RGB-D occlusion, full-CAD targets, memory TBPTT."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from torch import nn
from lip.data.jepa_rigid_fast import FastRigidEpisodeFactory
from lip.data.jepa_rigid_episodes import sample_center_perturbation
from lip.geometry.so3 import center_pose
from lip.losses import pose_loss
from .features import prepare_scene,encode_scenes,build_teachers
from .losses import reconstruction_loss,RECONSTRUCTION_METRICS
from .occlusion import OccluderBank
from .renderer import FullTextureRenderer
from .timing import Timings


@dataclass
class Episode:
    rgb: torch.Tensor
    depth: torch.Tensor
    initial: torch.Tensor
    mesh: dict
    k: torch.Tensor
    times: list
    stream: str
    cad: dict
    occlusion_plan: object
    history: bool
    cad_enabled: bool
    training_window: dict|None=None


class Factory(FastRigidEpisodeFactory):
    def __init__(self,config,model,store,split='train'):
        p=config['paths']
        super().__init__(p['data_root'],p['index_root'],model.encoder,split=split,
                         packed_root=p['rigid_frame_cache'] if split=='train' else None)
        self.config=config;self.store=store
        self.renderer=FullTextureRenderer(self.device)
        self.occluders=OccluderBank(p['occluder_bank'],self.audit['split_hash'],self.device)
        if self.occluders.receipt['bank_sha256']!=config['augmentation']['bank_sha256']:
            raise ValueError('Configured occluder bank identity mismatch')

    def sample(self,seed,frames=None):
        from scipy.spatial.transform import Rotation
        t=self.config['training'];frames=frames or t['episode_frames']
        window=None
        if self.config.get('training_window',{}).get('enabled',False):
            from .full_window import prepare_cpu
            data,window=prepare_cpu(self,seed,frames)
            sid,initial,first,count,rng,arrays=data
        else:sid,initial,first,count,rng,arrays=self.prepare_cpu(seed,supervised=frames,burn=0)
        stream=self.streams[sid];mesh=self.mesh(stream)
        cad=self.store.get(self.root/stream['mesh_path'],mesh)
        rgb=self.upload(arrays[0]).float()/255;depth=self.upload(arrays[1])*self.audit['depth_scale_to_m']
        center=torch.as_tensor(mesh['center'],device=self.device)
        pose=center_pose(torch.tensor(initial['pose_original'],device=self.device),center)
        truth=center_pose(self.upload(arrays[3]),center)
        if rng.uniform()<t['perturb_probability']:
            displacement,rotation,_=sample_center_perturbation(rng,float(mesh['diameter']))
            pose=pose.clone();pose[:3,:3]=torch.tensor(Rotation.from_rotvec(rotation).as_matrix(),device=self.device,dtype=torch.float32)@pose[:3,:3]
            pose[:3,3]+=torch.tensor(displacement,device=self.device,dtype=torch.float32)
        if window is not None and first!=initial['frame_index']:
            from .full_window import transport_centered_error
            source_truth=center_pose(torch.as_tensor(self.pose_labels[sid][initial['frame_index']],device=self.device),center)
            pose=transport_centered_error(pose,source_truth,truth[0])
        heavy=bool(rng.integers(2));duration=int(rng.choice(t['heavy_durations']))
        plan=self.occluders.plan(rng,stream['object_id'],heavy=heavy,start=t['anchor_frames'],duration=duration)
        return Episode(rgb,depth,pose,mesh,torch.tensor(stream['intrinsics'],device=self.device),
            [frame/self.audit['fps'] for frame in range(first,first+count)],sid+f'|draw{seed}',cad,plan,
            heavy or bool(rng.integers(2)),not(heavy and rng.uniform()<t['heavy_cad_dropout']),window),(truth,self.upload(arrays[2]))


class TrainingEpisode(nn.Module):
    def __init__(self,model,renderer,config):
        super().__init__();self.model=model;self.renderer=renderer;self.config=config;self.timings={}

    def forward(self,episodes,targets):
        self.profile=Timings();self.model.profile_timing=self.profile
        model=self.model;device=episodes[0].rgb.device;b=len(episodes)
        poses=torch.stack([e.initial for e in episodes]);previous=None;memory=None
        gt=torch.stack([t[0] for t in targets]);visible=torch.stack([t[1] for t in targets])
        points=torch.stack([torch.as_tensor(e.mesh['points'],device=device) for e in episodes])
        diameter=torch.tensor([float(e.mesh['diameter']) for e in episodes],device=device)
        history=torch.tensor([e.history for e in episodes],device=device)
        cad_enabled=torch.tensor([e.cad_enabled for e in episodes],device=device)
        losses=[];parts=[];times={k:0. for k in ('crop_render','occlusion','encoders','teacher','jepa_loss')}
        coverage=[];donor_kinds=set();unrepresentable=0
        count=len(episodes[0].times)
        for frame in range(count):
            begun=time.monotonic()
            with self.profile.record('crop_and_student_render'):
                scenes=[prepare_scene(e.rgb[frame],e.depth[frame],poses[i],e.mesh,e.k,e.times[frame],e.stream,e.cad,
                self.renderer,None if previous is None else previous[i],None if frame==0 else e.times[frame-1]) for i,e in enumerate(episodes)]
            times['crop_render']+=time.monotonic()-begun;begun=time.monotonic()
            with self.profile.record('textured_rgbd_occlusion'):
                occlusions=[e.occlusion_plan.render(s,frame) for e,s in zip(episodes,scenes)]
            times['occlusion']+=time.monotonic()-begun;begun=time.monotonic()
            masks=[o.mask for o in occlusions]
            for s,o in zip(scenes,occlusions):
                if o.provenance:
                    silhouette=s.render['mask'][None,None]&s.bounds
                    coverage.append(float((o.mask&silhouette).sum()/silhouette.sum().clamp_min(1)))
                    donor_kinds.update(x[0] for x in o.provenance)
            obs=encode_scenes(model,scenes,cad_enabled=cad_enabled,frame_id=frame,occlusions=occlusions)
            unrepresentable+=len(model.utonia.last_unrepresentable_clouds)
            times['encoders']+=time.monotonic()-begun;begun=time.monotonic()
            if frame:
                with self.profile.record('fullcad_rgbd_teacher'):
                    teacher=build_teachers(model.encoder,scenes,gt[:,frame],visible[:,frame],masks,self.renderer)
            times['teacher']+=time.monotonic()-begun;begun=time.monotonic()
            # Preserve gradients through frozen predictor operations to new input
            # projections/writer, including initialization of the first memory.
            with self.profile.record('jepa_and_pose'):
                output,memory=model(obs,memory,history)
            if frame:
                pose,pose_parts=pose_loss(output['pose_centered'],gt[:,frame],points,diameter)
                reconstruction,rec_parts=reconstruction_loss(output,teacher,self.config['training']['loss_weights'])
                losses.append(pose+reconstruction)
                parts.append(torch.stack([pose.detach(),pose_parts['translation'].detach(),pose_parts['rotation'].detach(),
                    pose_parts['points'].detach(),*[rec_parts[k] for k in RECONSTRUCTION_METRICS]]))
                previous=poses;poses=output['pose_centered'].detach()
            if (frame+1)%self.config['training']['tbptt_frames']==0:memory=memory.detach()
            times['jepa_loss']+=time.monotonic()-begun
        self.timings=times;self.model.profile_timing=None
        self.occlusion_diagnostics=dict(active_frames=len(coverage),mean_base_silhouette_coverage=float(np.mean(coverage)) if coverage else 0.,donor_kinds=sorted(donor_kinds),unrepresentable_geometry_clouds=unrepresentable)
        return torch.stack(losses).mean(),torch.stack(parts).mean(0)
