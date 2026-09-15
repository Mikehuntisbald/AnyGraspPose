"""Local memory readout for a separate aligned-tracker architecture.

Coordinates are weighted means of predicted-pose rendered CAD coordinates,
normalized by object diameter. They are soft alignment hints, not established
pixel correspondences or camera-frame points. No GT or hand labels enter here.
"""
import torch
from torch import nn
from torch.nn import functional as F


def pool_memory_geometry(geometry,side):
    """Return B,N,5: mean XYZ/d, coordinate variance, silhouette coverage."""
    if geometry.ndim!=4 or geometry.shape[1]!=9 or side<1:raise ValueError('Expected nine-channel geometry and positive grid side')
    g=geometry.float();silhouette=(g[:,3:4]>.5).float();xyz=g[:,4:7]
    coverage=F.adaptive_avg_pool2d(silhouette,side)
    mean=F.adaptive_avg_pool2d(xyz*silhouette,side)/coverage.clamp_min(1e-8)
    second=F.adaptive_avg_pool2d(xyz.square().sum(1,keepdim=True)*silhouette,side)/coverage.clamp_min(1e-8)
    variance=(second-mean.square().sum(1,keepdim=True)).clamp_min(0)
    mean=torch.where(coverage>0,mean,0.);variance=torch.where(coverage>0,variance,0.)
    return torch.cat((mean,variance,coverage),1).flatten(2).transpose(1,2)


def canonical_distance(query_geometry,key_geometry,sigma=.05):
    """Dimensionless soft distance, accounting for spatial pooling variance."""
    if sigma<=0:raise ValueError('Positive coordinate bandwidth required')
    q=query_geometry.float();k=key_geometry.float()
    distance=(q[:,:,None,:3]-k[:,None,:,:3]).square().sum(-1)
    spread=(q[:,:,None,3]+k[:,None,:,3]+sigma**2).clamp_min(1e-8)
    return (distance/spread).clamp_max(25.)


class AlignedMemoryReadout(nn.Module):
    """Current local queries read causal, geometrically related keyframe tokens."""
    def __init__(self,recent_frames=8,max_age=64,sigma=.05):
        super().__init__()
        if recent_frames<1 or max_age<recent_frames or sigma<=0:raise ValueError('Invalid alignment memory contract')
        self.recent_frames=recent_frames;self.max_age=max_age;self.sigma=sigma
        self.query_norm=nn.LayerNorm(256);self.key_norm=nn.LayerNorm(256)
        self.attention=nn.MultiheadAttention(256,8,dropout=0.,batch_first=True)
        self.geometry_strength=nn.Parameter(torch.zeros(8))
        self.gate=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,1))
        nn.init.zeros_(self.attention.out_proj.weight);nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-2.)

    def forward(self,object_latent,current_tokens,current_geometry,memory_tokens,memory_geometry,bank,meta,key_condition=None):
        if current_tokens.ndim!=3 or memory_tokens.ndim!=4:raise ValueError('Expected local queries and per-slot memory tokens')
        b,slots,n,width=memory_tokens.shape;q=current_tokens.shape[1]
        if min(b,slots,n,q)<1:raise ValueError('Nonempty batch, queries and cache slots required')
        if width!=256 or current_tokens.shape!=(b,q,256) or object_latent.shape!=(b,256):raise ValueError('256-dimensional latent contract required')
        if current_geometry.shape!=(b,q,5) or memory_geometry.shape!=(b,slots,n,5) or bank.valid.shape!=(b,slots):raise ValueError('Geometry and cache identity shapes differ')
        age=meta.frame_id[:,None]-bank.frame_id
        slot_valid=bank.valid&(age>=self.recent_frames)&(age<=self.max_age)
        slot_valid&=(bank.timestamp<meta.timestamp[:,None])&(bank.stream_tag==meta.stream_tag[:,None])
        slot_valid&=meta.key_valid.any(-1)[:,None]
        key_geometry=memory_geometry.flatten(1,2).float()
        allowed=slot_valid.repeat_interleave(n,1)&(key_geometry[...,4]>0)
        has_keys=allowed.any(1)
        key_geometry=torch.where(allowed[...,None],key_geometry,0.)
        context=memory_tokens
        if key_condition is not None:
            if key_condition.shape!=(b,slots,256):raise ValueError('Expected one condition per keyframe')
            context=context+key_condition[:,:,None]
        context=context.flatten(1,2);context=torch.where(allowed[...,None],context,0.)
        query_geometry=current_geometry.float();query_weight=query_geometry[...,4].clamp(0,1)
        query_weight=query_weight*meta.key_valid.any(-1)[:,None]
        query_geometry=torch.where((query_weight>0)[...,None],query_geometry,0.)
        distance=canonical_distance(query_geometry,key_geometry,self.sigma)
        strength=F.softplus(self.geometry_strength.float())
        bias=-strength[None,:,None,None]*distance[:,None]
        bias=bias+key_geometry[...,4].clamp_min(1e-6).log()[:,None,None,:]
        fallback=torch.zeros_like(allowed);fallback[:,0]=~has_keys
        bias=bias.masked_fill(~(allowed|fallback)[:,None,None,:],float('-inf'))
        query=self.query_norm(current_tokens+object_latent[:,None]);context=self.key_norm(context)
        h,_=self.attention(query,context,context,attn_mask=bias.reshape(b*8,q,slots*n).to(query.dtype),need_weights=False)
        pooled=(h*query_weight.to(h.dtype)[...,None]).sum(1)/query_weight.sum(1,keepdim=True).clamp_min(1e-8).to(h.dtype)
        usable=has_keys&(query_weight.sum(1)>0)
        pooled=torch.where(usable[:,None],pooled,0.)
        gate=torch.sigmoid(self.gate(torch.cat((object_latent,pooled),-1)))
        return gate*pooled,dict(usable_tokens=allowed.sum(1),usable_queries=(query_weight>0).sum(1),
            geometry_strength=strength.detach(),update_norm=(gate*pooled).float().norm(dim=-1))
