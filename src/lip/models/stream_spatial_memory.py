"""Causal dense keyframe features with pose/crop-conditioned gated readout."""
from dataclasses import dataclass,replace
import torch
from torch import nn
from torch.nn import functional as F
from lip.models.stream_rk import RKTracker,RKCache,AnchorBank,observation_support
from lip.models.tracker import spatial_position
from lip.engine.stream_state import CrossCache,TemporalCache
from lip.geometry.so3 import update,original_pose


def dense_source_xy(coarse_xy,side):
    """Exact affine extrapolation from 4x4 cell centers to dense cell centers."""
    grid=coarse_xy.reshape(-1,4,4,2)
    dx=grid[:,0,1]-grid[:,0,0];dy=grid[:,1,0]-grid[:,0,0]
    origin=grid[:,0,0]-.5*(dx+dy)
    cell=(torch.arange(side,device=coarse_xy.device,dtype=coarse_xy.dtype)+.5)*4/side
    yy,xx=torch.meshgrid(cell,cell,indexing='ij')
    return origin[:,None]+xx.flatten()[None,:,None]*dx[:,None]+yy.flatten()[None,:,None]*dy[:,None]


@dataclass(frozen=True)
class SpatialCache(RKCache):
    patches: torch.Tensor|None=None
    patch_quality: torch.Tensor|None=None

    def detach(self):
        return SpatialCache(self.source.detach(),tuple(x.detach() for x in self.contexts),
            anchors=None if self.anchors is None else self.anchors.detach(),variant=self.variant,
            patches=None if self.patches is None else self.patches.detach(),patch_quality=None if self.patch_quality is None else self.patch_quality.detach())

    @property
    def kv_bytes(self):
        return super().kv_bytes+sum(t.numel()*t.element_size() for t in (self.patches,self.patch_quality) if t is not None)


def select_patches(old_bank,next_bank,patches,quality,current,current_quality,meta):
    b,slots=old_bank.valid.shape;n,d=current.shape[1:]
    if patches is None:
        patches=current.new_zeros(b,slots,n,d);quality=current_quality.new_zeros(b,slots,n)
    if patches.shape!=(b,slots,n,d):raise ValueError('Dense token shape changed; reset the stream')
    frame=torch.cat((old_bank.frame_id,meta.frame_id[:,None]),1)
    tag=torch.cat((old_bank.stream_tag,meta.stream_tag[:,None]),1)
    valid=torch.cat((old_bank.valid,meta.key_valid.any(-1)[:,None]),1)
    match=(next_bank.frame_id[:,:,None]==frame[:,None])&(next_bank.stream_tag[:,:,None]==tag[:,None])&valid[:,None]
    index=match.long().argmax(-1)
    keep=next_bank.valid&match.any(-1)
    p=torch.cat((patches,current[:,None]),1).gather(1,index[:,:,None,None].expand(b,slots,n,d))
    q=torch.cat((quality,current_quality[:,None]),1).gather(1,index[:,:,None].expand(b,slots,n))
    return torch.where(keep[:,:,None,None],p,0.),torch.where(keep[:,:,None],q,0.)


class SpatialRKTracker(RKTracker):
    def __init__(self,memory_frames=8,dropout=0.,time_unit=1/30,max_gap_seconds=.5,slots=4,min_gap=4,max_age=64,tolerance=.05,dense_side=14):
        super().__init__(True,True,memory_frames,dropout,time_unit,max_gap_seconds,slots,min_gap,max_age,tolerance)
        self.architecture_id='stream_rk_spatial';self.cache_contract='lip-rk-spatial-memory-v1'
        self.dense_side=dense_side;self.variant=f'R1K1_spatial{dense_side}'
        self.patch_attention=nn.MultiheadAttention(256,8,dropout=0.,batch_first=True)
        nn.init.zeros_(self.patch_attention.out_proj.weight);nn.init.zeros_(self.patch_attention.out_proj.bias)
        self.patch_gate=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,1))
        nn.init.zeros_(self.patch_gate[-1].weight);nn.init.constant_(self.patch_gate[-1].bias,-2.)
        self.patch_pose=nn.Sequential(nn.Linear(14,128),nn.GELU(),nn.Linear(128,256))
        nn.init.zeros_(self.patch_pose[-1].weight);nn.init.zeros_(self.patch_pose[-1].bias)

    def encode_dense(self,features):
        rgb=features['rgb'];x=self.rgb_proj(self.rgb(rgb));g=self.geometry(features['geometry']);side=x.shape[-1]
        if side!=self.dense_side:raise ValueError('Unexpected encoder grid size')
        pos=spatial_position(side,x.device).to(x.dtype)
        x=x.flatten(2).transpose(1,2)+pos;g=g.flatten(2).transpose(1,2)+pos
        for block in self.fusion:x=block(x,g)
        dense=x+self.token_type[0]+self.source_position(dense_source_xy(features['source_xy'],side))
        x=F.adaptive_avg_pool2d(x.transpose(1,2).reshape(len(rgb),256,side,side),4).flatten(2).transpose(1,2)
        x=x+spatial_position(4,x.device).to(x.dtype)+self.token_type[0]+self.source_position(features['source_xy'])
        state=self.state(features['state_input'])+self.token_type[1]
        return torch.cat((x,state[:,None]),1),dense

    def pose_condition(self,bank,features,meta):
        base=features['T_base_centered'].float();anchor=bank.pose.float();diameter=features['object_diameter_m'].float()
        rel=base[:,None,:3,:3]@anchor[:,:,:3,:3].transpose(-1,-2)
        rotation=rel[:,:,:,:2].transpose(-1,-2).flatten(2)
        translation=(base[:,None,:3,3]-anchor[:,:,:3,3])/diameter[:,None,None]
        current_k=features['state_input'][:,10:14].float()*224
        focal=torch.log(current_k[:,None,:2].clamp_min(1)/bank.crop_intrinsics[:,:,:2].clamp_min(1))
        principal=(current_k[:,None,2:]-bank.crop_intrinsics[:,:,2:])/224
        elapsed=torch.log1p((meta.timestamp[:,None]-bank.timestamp).clamp_min(0)/self.temporal.time_unit).float()[...,None]
        return self.patch_pose(torch.cat((rotation,translation,focal,principal,elapsed),-1).clamp(-20,20))

    def patch_read(self,objects,bank,patches,quality,features,meta):
        age=meta.frame_id[:,None]-bank.frame_id
        valid=bank.valid&(age>=self.memory_frames)&(age<=self.max_age)&(bank.stream_tag==meta.stream_tag[:,None])&(bank.timestamp<=meta.timestamp[:,None])
        valid&=meta.key_valid.any(-1)[:,None]
        n=self.dense_side**2
        if patches is None:
            patches=objects.new_zeros(len(objects),self.slots,n,256);quality=objects.new_zeros(len(objects),self.slots,n)
            valid=valid&False
        condition=self.pose_condition(bank,features,meta)
        context=(torch.where(bank.valid[:,:,None,None],patches,0.)+condition[:,:,None]).flatten(1,2)
        allowed=valid.repeat_interleave(n,1);has=allowed.any(1)
        fallback=torch.zeros_like(allowed);fallback[:,0]=~has
        bias=quality.flatten(1).clamp_min(.25).log().masked_fill(~(allowed|fallback),float('-inf'))
        bias=bias[:,None].repeat_interleave(8,0).to(objects.dtype)
        h,_=self.patch_attention(objects[:,None],context,context,attn_mask=bias,need_weights=False)
        h=torch.where(has[:,None,None],h,0.)[:,0]
        gate=torch.sigmoid(self.patch_gate(torch.cat((objects,h),-1)))
        return gate*h,allowed.sum(1)

    def forward(self,features,metadata,cache=None,profiler=None):
        result,next_cache,_,_=self.forward_with_dense(features,metadata,cache,profiler)
        return result,next_cache

    def forward_with_dense(self,features,metadata,cache=None,profiler=None):
        """Same parent forward, with local features for optional residual readers."""
        if isinstance(cache,SpatialCache):
            if cache.variant!=self.variant:raise ValueError('Spatial cache configuration changed')
        elif cache is not None and (not isinstance(cache,CrossCache) or cache.metadata):raise ValueError('Spatial memory requires its own cache')
        cache=cache or CrossCache(TemporalCache(capacity=self.memory_frames))
        token_quality,quality=observation_support(features['geometry'],self.support_tolerance)
        extra=torch.cat((token_quality.log(),token_quality.new_zeros(len(quality),1)),1)
        metadata=replace(metadata,role_bias=metadata.role_bias+2*torch.tanh(self.reliability_strength)*extra)
        source,dense=self.encode_dense(features);query=self.readout.query.expand(len(source),-1,-1)+self.token_type[2]
        z,next_source=self.temporal(source,query,metadata,cache.source,True);result,contexts=self.readout(z,metadata,cache.contexts)
        bank=getattr(cache,'anchors',None) or AnchorBank.empty(z[:,1],self.slots)
        old_patches=getattr(cache,'patches',None);old_quality=getattr(cache,'patch_quality',None)
        residual,used=self.anchor_read(result['latent_object'][:,None],bank,metadata)
        result['latent']=result['latent']+residual
        patch_residual,patches_read=self.patch_read(result['latent_object'],bank,old_patches,old_quality,features,metadata)
        result['latent']=result['latent']+patch_residual
        next_bank=bank.insert(z[:,1],quality,metadata,features,self.min_gap,self.max_age)
        dense_quality,_=observation_support(features['geometry'],self.support_tolerance,self.dense_side)
        patches,patch_quality=select_patches(bank,next_bank,old_patches,old_quality,dense,dense_quality,metadata)
        delta=self.head(result['latent']).float();delta=delta*(1-.5*torch.tanh(self.update_strength)*(1-quality))[:,None]
        with torch.autocast(source.device.type,enabled=False):
            pose=update(features['T_base_centered'].float(),delta[:,:3],delta[:,3:],features['object_diameter_m'].float())
            result.update(pose_centered=pose,pose_original=original_pose(pose,features['mesh_center'].float()),delta_rotvec=delta[:,:3],delta_center_norm=delta[:,3:],
                confidence=None,observation_support=quality,anchors_read=used,anchor_norm=next_bank.latent.float().norm(dim=-1).mean(1),
                spatial_tokens_read=patches_read,spatial_update_norm=patch_residual.float().norm(dim=-1))
        return result,SpatialCache(next_source,contexts,anchors=next_bank,variant=self.variant,patches=patches,patch_quality=patch_quality),source,dense
