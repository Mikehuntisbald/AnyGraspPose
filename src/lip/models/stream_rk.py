"""Factorial observation weighting and causal quality-selected keyframe memory.

Geometry support is a proxy, not a calibrated probability. R controls whether
it modulates attention/corrections. K independently enables the same fixed
support-ranked anchor bank in both R conditions. No labels or FP enter forward.
"""
from dataclasses import dataclass, replace
import torch
from torch import nn
from torch.nn import functional as F
from lip.models.stream_tracker import StreamTracker
from lip.engine.stream_state import CrossCache, TemporalCache
from lip.geometry.so3 import update, original_pose


def observation_support(geometry, tolerance=.05, side=4):
    """Nine-channel geometry: valid=1, silhouette=3, residual/d=7, both=8."""
    g=geometry.float()
    silhouette=g[:,3:4]>.5
    both=(g[:,8:9]>.5)&silhouette
    residual=g[:,7:8]
    known=.25+.75*torch.exp(-residual.abs()/tolerance)
    quality=torch.where(both,known,torch.full_like(known,.5))
    count=F.adaptive_avg_pool2d(silhouette.float(),side)
    pooled=F.adaptive_avg_pool2d(quality*silhouette,side)
    token=torch.where(count>0,pooled/count.clamp_min(1e-6),.5).flatten(1)
    mass=silhouette.flatten(1).sum(1)
    frame=torch.where(mass>0,(quality*silhouette).flatten(1).sum(1)/mass.clamp_min(1),.5)
    return token.clamp(.25,1),frame.clamp(.25,1)


@dataclass(frozen=True)
class AnchorBank:
    latent: torch.Tensor
    quality: torch.Tensor
    frame_id: torch.Tensor
    timestamp: torch.Tensor
    stream_tag: torch.Tensor
    valid: torch.Tensor
    pose: torch.Tensor
    crop_intrinsics: torch.Tensor
    source_xy: torch.Tensor

    def detach(self):
        return AnchorBank(*(getattr(self,n).detach() for n in self.__dataclass_fields__))

    @property
    def bytes(self):
        return sum(getattr(self,n).numel()*getattr(self,n).element_size() for n in self.__dataclass_fields__)

    @classmethod
    def empty(cls,latent,slots):
        b=len(latent);device=latent.device
        return cls(latent.new_zeros(b,slots,256),torch.zeros(b,slots,device=device),
            torch.full((b,slots),-1,device=device,dtype=torch.long),
            torch.zeros(b,slots,device=device,dtype=torch.float64),
            torch.full((b,slots),-1,device=device,dtype=torch.long),
            torch.zeros(b,slots,device=device,dtype=torch.bool),
            torch.zeros(b,slots,4,4,device=device),torch.zeros(b,slots,4,device=device),
            torch.zeros(b,slots,16,2,device=device))

    def insert(self,latent,quality,meta,features,min_gap,max_age):
        """Greedy quality selection with temporal separation; no cross-lane mixing."""
        current=dict(latent=latent,quality=quality,frame_id=meta.frame_id,
            timestamp=meta.timestamp,stream_tag=meta.stream_tag,valid=meta.key_valid.any(-1),
            pose=features['T_base_centered'],crop_intrinsics=features['state_input'][:,10:14]*224,
            source_xy=features['source_xy'])
        values={n:torch.cat((getattr(self,n),current[n][:,None]),1) for n in current}
        age=meta.frame_id[:,None]-values['frame_id']
        valid=values['valid']&(values['stream_tag']==meta.stream_tag[:,None])&(age>=0)&(age<=max_age)
        valid &= values['timestamp']<=meta.timestamp[:,None]
        # Do not replace the newest anchor every frame: that would keep a bank
        # of only one frame on steady-quality video. Admit spaced frames, or
        # replace a nearby anchor only for a meaningful support improvement.
        nearby=valid[:,:-1]&(age[:,:-1]<min_gap)
        nearby_quality=values['quality'][:,:-1].masked_fill(~nearby,float('-inf')).max(1).values
        valid[:,-1] &= ~nearby.any(1)|(quality.detach()>nearby_quality.detach()+.05)
        # Detach discrete selection. Small age penalty replaces equally reliable stale entries.
        priority=(values['quality'].detach()-.001*age.float()).masked_fill(~valid,float('-inf'))
        indices=[];selected_valid=[]
        for _ in range(self.valid.shape[1]):
            index=priority.argmax(1);good=torch.isfinite(priority.gather(1,index[:,None])[:,0])
            indices.append(index);selected_valid.append(good)
            chosen=values['frame_id'].gather(1,index[:,None])
            close=(values['frame_id']-chosen).abs()<min_gap
            priority=priority.masked_fill(close,float('-inf'))
        index=torch.stack(indices,1);mask=torch.stack(selected_valid,1)
        result={}
        for name,value in values.items():
            shape=index.shape+value.shape[2:]
            idx=index.reshape(*index.shape,*([1]*(value.ndim-2))).expand(shape)
            selected=value.gather(1,idx)
            keep=mask.reshape(*mask.shape,*([1]*(value.ndim-2)))
            result[name]=torch.where(keep,selected,torch.zeros_like(selected))
        result['valid']=mask
        return AnchorBank(**result)


@dataclass(frozen=True)
class RKCache(CrossCache):
    anchors: AnchorBank|None=None
    variant: str=''

    def detach(self):
        return RKCache(self.source.detach(),tuple(x.detach() for x in self.contexts),
                       anchors=None if self.anchors is None else self.anchors.detach(),variant=self.variant)

    @property
    def kv_bytes(self):
        return super().kv_bytes+(0 if self.anchors is None else self.anchors.bytes)


class RKTracker(StreamTracker):
    def __init__(self,reliability=False,keyframes=False,memory_frames=8,dropout=0.,
                 time_unit=1/30,max_gap_seconds=.5,slots=4,min_gap=4,max_age=64,tolerance=.05):
        super().__init__('stream_dual_cross_residual',memory_frames,dropout,False,time_unit,max_gap_seconds)
        self.architecture_id='stream_rk_factorial'
        self.cache_contract='lip-rk-factorial-v1'
        self.reliability=bool(reliability);self.keyframes=bool(keyframes)
        self.variant=f'R{int(reliability)}K{int(keyframes)}'
        self.slots=slots;self.min_gap=min_gap;self.max_age=max_age;self.support_tolerance=tolerance
        self.reliability_strength=nn.Parameter(torch.zeros(()))
        self.update_strength=nn.Parameter(torch.zeros(()))
        self.anchor_attn=nn.MultiheadAttention(256,8,dropout=0.,batch_first=True)
        nn.init.zeros_(self.anchor_attn.out_proj.weight);nn.init.zeros_(self.anchor_attn.out_proj.bias)
        self.anchor_gate=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,1))
        nn.init.zeros_(self.anchor_gate[-1].weight);nn.init.constant_(self.anchor_gate[-1].bias,-2.)
        self.anchor_time=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,256))
        nn.init.zeros_(self.anchor_time[-1].weight);nn.init.zeros_(self.anchor_time[-1].bias)
        # All arms adapt the same existing cross branch; the established backbone stays fixed.
        for name,param in self.named_parameters():
            common=name.startswith(('readout.cross_attn.','readout.cross_gate.'))
            extra=(self.keyframes and name.startswith(('anchor_attn.','anchor_gate.','anchor_time.')))
            quality=self.reliability and name in ('reliability_strength','update_strength')
            param.requires_grad_(common or extra or quality)

    def anchor_read(self,objects,bank,meta):
        age=meta.frame_id[:,None]-bank.frame_id
        valid=bank.valid&(age>=self.memory_frames)&(age<=self.max_age)
        valid &= (bank.timestamp<=meta.timestamp[:,None])&(bank.stream_tag==meta.stream_tag[:,None])
        valid &= meta.key_valid.any(-1)[:,None]
        elapsed=(meta.timestamp[:,None]-bank.timestamp).clamp_min(0)
        temporal=self.anchor_time(torch.log1p(elapsed/self.temporal.time_unit).float()[...,None])
        context=torch.where(bank.valid[...,None],bank.latent,0.)+temporal
        any_valid=valid.any(1)
        fallback=torch.zeros_like(valid);fallback[:,0]=~any_valid
        bias=torch.zeros_like(bank.quality)
        if self.reliability:
            bias=bias+2*torch.tanh(self.reliability_strength)*bank.quality.clamp_min(.25).log()
        bias=bias.masked_fill(~(valid|fallback),float('-inf'))
        bias=bias[:,None,:].repeat_interleave(8,0).to(objects.dtype)
        h,_=self.anchor_attn(objects,context,context,attn_mask=bias,need_weights=False)
        h=torch.where(any_valid[:,None,None],h,0.)
        gate=torch.sigmoid(self.anchor_gate(torch.cat((objects,h),-1)))
        return (gate*h)[:,0],valid.sum(1)

    def forward(self,features,metadata,cache=None,profiler=None):
        if isinstance(cache,RKCache) and cache.variant!=self.variant:
            raise ValueError('Factorial cache belongs to a different arm')
        if cache is not None and not isinstance(cache,CrossCache):
            raise ValueError('RK model requires a cross cache')
        cache=cache or CrossCache(TemporalCache(capacity=self.memory_frames))
        token_quality,quality=observation_support(features['geometry'],self.support_tolerance)
        if self.reliability:
            extra=torch.cat((token_quality.log(),token_quality.new_zeros(len(quality),1)),1)
            metadata=replace(metadata,role_bias=metadata.role_bias+2*torch.tanh(self.reliability_strength)*extra)
        source=self.encode_current(features,profiler)
        query=self.readout.query.expand(len(source),-1,-1)+self.token_type[2]
        z,next_source=self.temporal(source,query,metadata,cache.source,True)
        result,contexts=self.readout(z,metadata,cache.contexts)
        bank=getattr(cache,'anchors',None)
        used=torch.zeros(len(source),device=source.device,dtype=torch.long)
        if self.keyframes:
            bank=bank or AnchorBank.empty(z[:,1],self.slots)
            residual,used=self.anchor_read(result['latent_object'][:,None],bank,metadata)
            result['latent']=result['latent']+residual
            # Current context is written only AFTER querying strictly older anchors.
            bank=bank.insert(z[:,1],quality,metadata,features,self.min_gap,self.max_age)
            result['anchor_norm']=bank.latent.float().norm(dim=-1).mean(1)
        delta=self.head(result['latent']).float()
        if self.reliability:
            # Smooth, bounded [0.5,1.5] modulation can learn to damp or amplify;
            # geometry disagreement never hard-stops correction from a poor prior.
            delta=delta*(1-.5*torch.tanh(self.update_strength)*(1-quality))[:,None]
        with torch.autocast(source.device.type,enabled=False):
            pose=update(features['T_base_centered'].float(),delta[:,:3],delta[:,3:],features['object_diameter_m'].float())
            result.update(pose_centered=pose,pose_original=original_pose(pose,features['mesh_center'].float()),
                delta_rotvec=delta[:,:3],delta_center_norm=delta[:,3:],confidence=None,
                observation_support=quality,anchors_read=used)
        return result,RKCache(next_source,contexts,anchors=bank,variant=self.variant)
