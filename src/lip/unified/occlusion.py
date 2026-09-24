"""Temporally coherent train-only real hand/object cutouts with paired RGB-D."""
from collections import OrderedDict
from dataclasses import dataclass,field
import json,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lip.engine.jepa_checkpoint import sha


@dataclass(frozen=True)
class Occlusion:
    rgb: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor
    provenance: tuple=()


@dataclass
class OcclusionPlan:
    layers: tuple
    start: int
    duration: int
    target_fraction: float
    transforms: tuple
    calibration: tuple|None=field(default=None,init=False)

    @torch.no_grad()
    def _layers(self,scene,frame,center,extent,scale):
        y,x=torch.meshgrid(torch.arange(224,device=scene.rgb.device,dtype=torch.float32),torch.arange(224,device=scene.rgb.device,dtype=torch.float32),indexing='ij')
        result=[];t=frame-self.start
        for donor,param in zip(self.layers,self.transforms):
            dx,dy,vx,vy,rotation,angular,relative=param
            theta=rotation+angular*t;co,si=math.cos(theta),math.sin(theta)
            h,w=donor['alpha'].shape[-2:];side=extent*scale*relative
            width=side*w/max(h,w);height=side*h/max(h,w)
            px=x-(center[0]+dx*extent+vx*t);py=y-(center[1]+dy*extent+vy*t)
            grid=torch.stack((2*(co*px+si*py)/width,2*(-si*px+co*py)/height),-1)[None]
            rgb=F.grid_sample(donor['rgb'][None],grid,align_corners=False,padding_mode='border')
            other=torch.cat((donor['alpha'].float(),donor['relative_depth'],donor['depth_valid'].float()))[None]
            sampled=F.grid_sample(other,grid,align_corners=False,mode='nearest')
            alpha=(sampled[:,:1]>.5)&scene.bounds
            result.append((rgb,sampled[:,1:2],sampled[:,2:3]>.5,alpha))
        return result

    @torch.no_grad()
    def render(self,scene,frame):
        if not self.start<=frame<self.start+self.duration:
            return Occlusion(scene.rgb,scene.depth,torch.zeros_like(scene.bounds))
        if self.calibration is None:
            support=scene.render['mask']&scene.bounds[0,0]
            positions=support.nonzero()
            if len(positions):
                low=positions.amin(0);high=positions.amax(0);mid=(low+high).float()/2
                center=(float(mid[1]),float(mid[0]));extent=max(28.,float((high-low).max()))
            else:center=(111.5,111.5);extent=84.
            lo,hi=.15,6.
            for _ in range(9):
                scale=(lo+hi)/2;layers=self._layers(scene,self.start,center,extent,scale)
                alpha=torch.stack([x[3] for x in layers]).any(0)
                fraction=float((alpha[0,0]&support).sum()/support.sum().clamp_min(1))
                if fraction<self.target_fraction:lo=scale
                else:hi=scale
            self.calibration=(center,extent,hi)
        center,extent,scale=self.calibration;rgb=scene.rgb.clone();depth=scene.depth.clone();mask=torch.zeros_like(scene.bounds)
        metadata=[]
        for donor,(color,relative,valid,alpha) in zip(self.layers,self._layers(scene,frame,center,extent,scale)):
            # A rigid donor-depth offset places its measured surface in front of
            # all supported destination depths; missing donor depths stay missing.
            destination=depth[alpha&(depth>0)]
            front=destination.min() if destination.numel() else scene.pose[2,3]
            margin=max(.01,.08*scene.diameter)
            relief=relative.clamp(-.08,.08)
            maximum=relief[alpha&valid].max() if (alpha&valid).any() else relief.new_tensor(0.)
            donor_depth=(front-margin-maximum+relief).clamp_min(.005)
            donor_depth=torch.where(valid,donor_depth,0.)
            rgb=torch.where(alpha,color,rgb);depth=torch.where(alpha,donor_depth,depth);mask|=alpha
            metadata.append((donor['kind'],donor['object_id'],donor['source_dir'],donor['frame']))
        return Occlusion(rgb,depth,mask,tuple(metadata))


class OccluderBank:
    def __init__(self,root,split_hash,device='cuda'):
        self.root=Path(root);self.device=device;self.receipt=json.loads((self.root/'receipt.json').read_text())
        r=self.receipt
        if not r['completed'] or r['split']!='train' or r['split_hash']!=split_hash or r['official_test_access'] or r['validation_donors']:
            raise ValueError('Occluder train provenance mismatch')
        if sha(self.root/'bank.pt')!=r['bank_sha256']:raise ValueError('Occluder bank hash mismatch')
        self.items=torch.load(self.root/'bank.pt',map_location='cpu',weights_only=True)['cutouts']
        if any(x['source_split']!='train' for x in self.items):raise ValueError('Held-out occlusion donor')
        self.resident=OrderedDict()

    def get(self,index):
        if index not in self.resident:
            donor=dict(self.items[index])
            for key in ('rgb','alpha','relative_depth','depth_valid'):donor[key]=donor[key].to(self.device)
            donor['rgb']=donor['rgb'].float()/255
            from .features import DEPTH_SATURATION_M
            donor['depth_valid']=donor['depth_valid']&((donor['relative_depth']+donor['median_depth_m'])<DEPTH_SATURATION_M)
            self.resident[index]=donor
            while len(self.resident)>24:self.resident.popitem(last=False)
        self.resident.move_to_end(index);return self.resident[index]

    def plan(self,rng,object_id,*,heavy,start=8,duration=32,target_fraction=None):
        count=int(rng.integers(1,3)) if heavy else 1
        first='hand' if rng.uniform()<.5 else 'object';kinds=[first]+(['object' if first=='hand' else 'hand'] if count==2 else [])
        layers=[];transforms=[]
        for kind in kinds:
            eligible=[i for i,x in enumerate(self.items) if x['kind']==kind and x['object_id']!=object_id]
            if not eligible:raise ValueError('No eligible different-object donor')
            layers.append(self.get(int(rng.choice(eligible))))
            transforms.append((float(rng.uniform(-.2,.2)),float(rng.uniform(-.2,.2)),float(rng.uniform(-.22,.22)),
                float(rng.uniform(-.22,.22)),float(rng.uniform(-.7,.7)),float(rng.uniform(-.003,.003)),float(rng.uniform(.85,1.15))))
        fraction=float(rng.uniform(.75,.95) if heavy else rng.uniform(.1,.3)) if target_fraction is None else float(target_fraction)
        return OcclusionPlan(tuple(layers),start,duration,fraction,tuple(transforms))
