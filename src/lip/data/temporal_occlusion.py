"""Deterministic moving RGB-D occluders, with no current/future GT inputs."""
from dataclasses import dataclass,replace
import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class OccluderPlan:
    texture: torch.Tensor
    center: tuple[float,float]
    size: tuple[float,float]
    velocity: tuple[float,float]
    depth_m: float
    start: int
    end: int
    ellipse: bool


@torch.no_grad()
def make_plan(first_rgb,initial_pose,k,vertices,diameter,seed,total_frames,burn_in,probability=.5,startup_probability=0.):
    """Only the first observation and supplied noisy initial prior define a plan."""
    if not 0<=probability<=1:raise ValueError('Occlusion probability must be in [0, 1]')
    if not 0<=startup_probability<=1:raise ValueError('Startup probability must be in [0, 1]')
    if probability==0:return None
    if total_frames<burn_in+14:raise ValueError('Temporal occlusion needs at least 14 supervised frames')
    rng=np.random.default_rng(int(seed)^0x51C1A);h,w=first_rgb.shape[-2:]
    if rng.random()>=probability:return None
    camera=vertices.float()@initial_pose[:3,:3].float().T+initial_pose[:3,3].float()
    if not torch.isfinite(camera).all() or (camera[:,2]<=.01).any():return None
    uv=camera@k.float().T;xy=uv[:,:2]/uv[:,2:].clamp_min(.01)
    lo=xy.min(0).values.cpu().numpy();hi=xy.max(0).values.cpu().numpy();center=(lo+hi)/2
    box=np.maximum(hi-lo,8.)
    size=np.clip(box*rng.uniform(.9,1.5,2),[12.,12.],[.75*w,.75*h])
    center=center+rng.uniform(-.2,.2,2)*box
    # Default timing leaves burn-in unaugmented and four recovery frames. Raw
    # burn-in is not necessarily clear; it can contain natural occlusion.
    start=int(rng.integers(burn_in+2,max(burn_in+3,total_frames-19)))
    duration=int(rng.integers(16,33));end=min(total_frames-4,start+duration)
    velocity=rng.uniform(-.3,.3,2)*box/max(1,end-start)
    # A distant image corner supplies a real texture, not a class/hand label.
    tw=max(8,w//5);th=max(8,h//5);sx=0 if center[0]>w/2 else w-tw;sy=0 if center[1]>h/2 else h-th
    texture=first_rgb[:,sy:sy+th,sx:sx+tw].float().clone()
    plane=max(.05,float(initial_pose[2,3])-float(diameter)*float(rng.uniform(.65,.95)))
    plan=OccluderPlan(texture,tuple(map(float,center)),tuple(map(float,size)),tuple(map(float,velocity)),plane,start,end,bool(rng.integers(2)))
    # Independent branch RNG preserves appearance, motion, duration and the
    # existing random sequence. With zero probability the old plan is exact.
    if startup_probability and np.random.default_rng(np.random.SeedSequence([int(seed),0xC01D57])).random()<startup_probability:
        plan=replace(plan,start=0,end=plan.end-plan.start)
    return plan


@torch.no_grad()
def composite(rgb,depth,plan,index):
    """Opaque depth-tested overlay; closer real surfaces retain RGB and depth."""
    if plan is None or not plan.start<=index<plan.end:return rgb,depth,torch.zeros_like(depth,dtype=torch.bool)
    h,w=rgb.shape[-2:];elapsed=index-plan.start;cx=plan.center[0]+elapsed*plan.velocity[0];cy=plan.center[1]+elapsed*plan.velocity[1]
    yy,xx=torch.meshgrid(torch.arange(h,device=rgb.device,dtype=torch.float32),torch.arange(w,device=rgb.device,dtype=torch.float32),indexing='ij')
    u=(xx-cx)/(plan.size[0]/2);v=(yy-cy)/(plan.size[1]/2)
    inside=(u.square()+v.square()<=1) if plan.ellipse else ((u.abs()<=1)&(v.abs()<=1))
    valid_depth=torch.isfinite(depth)&(depth>0)
    mask=inside[None]&(~valid_depth|(depth>plan.depth_m))
    texture=F.grid_sample(plan.texture[None].to(rgb.device),torch.stack((u,v),-1)[None],mode='bilinear',padding_mode='border',align_corners=False)[0]
    return torch.where(mask,texture,rgb),torch.where(mask,depth.new_tensor(plan.depth_m),depth),mask
