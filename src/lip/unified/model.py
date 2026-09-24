"""One predictive temporal backbone; observed context and completed object routes."""
from dataclasses import dataclass,replace
import copy
import torch
from torch import nn
from torch.nn import functional as F
from lip.jepa.model import DinoJEPA
from lip.jepa.predictor import SafeAttention
from lip.jepa.positions import source_coordinates
from lip.jepa.evidence import EvidenceHead
from lip.models.stream_cross_readout import StreamResidualCrossReadout
from lip.geometry.so3 import update,original_pose


def local_pool(value,weight):
    b,_,d=value.shape
    den=weight.reshape(b,4,4,4,4).mean((2,4)).reshape(b,16)
    num=(value*weight[...,None]).reshape(b,4,4,4,4,d).mean((2,4)).reshape(b,16,d)
    return num/den.clamp_min(1e-6)[...,None],den


class UnifiedEvidenceHead(EvidenceHead):
    def forward(self,h,position,prompt,valid,past_summary=None):
        if past_summary is None:past_summary=torch.zeros_like(prompt)
        logits=self.head(torch.cat((h,prompt[:,None].expand_as(h),past_summary[:,None].expand_as(h)),-1)).squeeze(-1)
        p=logits.sigmoid()*valid
        context,mass=local_pool(h+position,valid.float())
        current=torch.cat((h+position+self.types[0],self.context_gate.sigmoid()*context+self.types[1]),1)
        bias=torch.cat((p.clamp_min(.05).log(),p.new_zeros(len(p),16)),1)
        return logits,p,current,torch.cat((valid,mass>0),1),bias


@dataclass(frozen=True)
class Record:
    features: torch.Tensor
    valid: torch.Tensor
    xy: torch.Tensor
    xyz: torch.Tensor
    xyz_valid: torch.Tensor
    timestamp: torch.Tensor
    quality: torch.Tensor

    def detach(self):return Record(**{k:getattr(self,k).detach() for k in self.__dataclass_fields__})


@dataclass(frozen=True)
class Memory:
    stream: tuple
    version: str
    objects: tuple=()
    contexts: tuple=()
    timestamp: torch.Tensor|None=None

    def detach(self):return replace(self,objects=tuple(r.detach() for r in self.objects),
        contexts=tuple(r.detach() for r in self.contexts),timestamp=None if self.timestamp is None else self.timestamp.detach())


class SpatialWriter(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(256,256)
        nn.init.eye_(self.proj.weight);nn.init.zeros_(self.proj.bias)
        self.ff=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,512),nn.GELU(),nn.Linear(512,256))
        nn.init.zeros_(self.ff[-1].weight);nn.init.zeros_(self.ff[-1].bias)

    def object_weight(self, visible, valid):
        return visible*valid

    def object_reliable(self, mass):
        return mass>=.5

    def forward(self,h,position,visible,valid,observation,memory):
        source=self.proj(h+position);source=source+self.ff(source)
        xy=source_coordinates(observation.packet.A_image_to_crop,observation.packet.source_size_wh)
        pool=local_pool
        def record(weight,object_record):
            features,mass=pool(source,weight)
            coordinates,_=pool(xy,weight);xyz,depth_mass=pool(observation.object_xyz,weight*observation.depth_valid)
            reliable=self.object_reliable(mass) if object_record else (mass>0)
            return Record(features,reliable,coordinates,xyz,depth_mass>0,observation.packet.timestamp_s,
                          (visible*valid).sum(-1)/valid.sum(-1).clamp_min(1))
        obj=record(self.object_weight(visible,valid),True);ctx=record(valid.float(),False)
        return self.commit(obj,ctx,observation,memory)

    def commit(self,obj,ctx,observation,memory):
        objects=list(memory.objects)
        if len(objects)<4:objects.append(obj)
        else:
            # Protect the first reliable per-lane anchor; replace only the other slots.
            def merge(old,new,choose):
                return Record(**{k:torch.where(choose.reshape(-1,*([1]*(getattr(old,k).ndim-1))),getattr(new,k),getattr(old,k)) for k in old.__dataclass_fields__})
            write=obj.valid.any(-1);empty=~objects[0].valid.any(-1)
            objects[0]=merge(objects[0],obj,write&empty)
            quality=torch.stack([r.quality.masked_fill(~r.valid.any(-1),-1) for r in objects[1:]],1)
            index=quality.argmin(1)
            for i in range(1,4):objects[i]=merge(objects[i],obj,write&~empty&(index==i-1))
        return Memory(memory.stream,memory.version,tuple(objects),(memory.contexts+(ctx,))[-4:],observation.packet.timestamp_s)


class UnifiedTracker(nn.Module):
    architecture_id='stream_dino_utonia_jepa_rgbd_v2'
    cache_contract='observed-local128-dino-utonia-rgbd-v2'

    def __init__(self,encoder,utonia,parent):
        super().__init__();self.encoder=encoder;self.utonia=utonia;self.weights_version='initial'
        self.core=DinoJEPA(None,memory_enabled=True,geometry_enabled=False)
        self.core.evidence=UnifiedEvidenceHead()
        self.core.writer=nn.Identity()  # No unused legacy writer parameters.
        transferred={k[5:]:v for k,v in parent.items() if k.startswith('core.') and not k.startswith(('core.writer.','core.geometry_adapter.'))}
        missing,extra=self.core.load_state_dict(transferred,strict=False)
        if missing or extra:raise ValueError(f'Predictor migration mismatch: {missing}, {extra}')
        self.geo_proj=nn.Linear(utonia.feature_dim,256);self.geo_position=nn.Linear(6,256)
        self.geo_type=nn.Embedding(2,256);self.cad_type=nn.Parameter(torch.zeros(256))
        self.cad_attn=nn.ModuleList([SafeAttention() for _ in range(4)])
        self.cad_norm=nn.ModuleList([nn.LayerNorm(256) for _ in range(4)])
        self.object_attn=nn.ModuleList([SafeAttention() for _ in range(4)])
        self.context_attn=nn.ModuleList([SafeAttention() for _ in range(4)])
        self.object_norm=nn.ModuleList([nn.LayerNorm(256) for _ in range(4)])
        self.context_norm=nn.ModuleList([nn.LayerNorm(256) for _ in range(4)])
        for attention,block in zip(self.cad_attn,self.core.blocks):
            attention.load_state_dict(block.obs.state_dict())
            nn.init.zeros_(attention.out.weight);nn.init.zeros_(attention.out.bias)
        self.state=nn.Sequential(nn.Linear(24,256),nn.GELU(),nn.Linear(256,256))
        self.readout=StreamResidualCrossReadout(8)
        self.head=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,6))
        for name,module in [('state',self.state),('readout',self.readout),('head',self.head)]:
            module.load_state_dict({k[len('lip.'+name+'.'):]:v for k,v in parent.items() if k.startswith('lip.'+name+'.')},strict=True)
        self.writer=SpatialWriter();self.memory_position=nn.Linear(6,256)
        nn.init.zeros_(self.memory_position.weight);nn.init.zeros_(self.memory_position.bias)
        # Each 14x14 pixel block is decoded from its shared 256-D patch latent.
        # XYZ is centered object-coordinate / diameter; depth is relative to base z.
        self.surface_head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,14*14*5))

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();self.utonia.eval();return self

    def empty_memory(self,observation):return Memory(observation.packet.stream_id,self.weights_version)

    def forward(self,obs,memory=None,history_enabled=None):
        packet=obs.packet;packet.validate();b=len(obs.mid)
        memory=self.empty_memory(obs) if memory is None else memory
        if memory.stream!=packet.stream_id or memory.version!=self.weights_version:raise ValueError('Stale/cross-stream memory')
        if memory.timestamp is not None and not torch.all(packet.timestamp_s>memory.timestamp):raise ValueError('Noncausal memory')
        enabled=torch.ones(b,device=obs.mid.device,dtype=torch.bool) if history_enabled is None else history_enabled
        h=self.core.src_proj(torch.cat((obs.mid,obs.last),-1))
        position,prompt,dt,xy=self.core.positions(packet,memory.timestamp)
        valid=F.avg_pool2d(packet.pixel_valid.float(),14,14).flatten(1)>=.999
        records=memory.objects+memory.contexts;mem=mv=mb=None;ctxmem=None;cmv=None
        past=None
        if records:
            feature=[];masks=[];bias=[]
            for i,r in enumerate(records):
                kind=0 if i<len(memory.objects) else 1
                spatial=torch.cat((r.xyz*r.xyz_valid[...,None],r.xy,r.xyz_valid[...,None].float()),-1)
                feature.append(r.features+self.core.memory_types.weight[kind]+self.memory_position(spatial))
                age=packet.timestamp_s-r.timestamp
                masks.append(r.valid&(age>0)[:,None]&enabled[:,None])
                bias.append(-F.softplus(self.core.memory_age_rate)*age.float().log1p()[:,None].expand(-1,16))
            mem=torch.cat(feature,1);mv=torch.cat(masks,1);mb=torch.cat(bias,1)
            n=len(memory.objects)*16
            if n:past=(mem[:,:n]*mv[:,:n,None]).sum(1)/mv[:,:n].sum(-1).clamp_min(1)[:,None]
            ctxmem=mem[:,n:];cmv=mv[:,n:]
        logits,p,current,cv,cb=self.core.evidence(h,position,prompt,valid,past)
        cad=self.core.src_proj(torch.cat((obs.cad_mid,obs.cad_last),-1))+position+self.cad_type
        geometry_features=self.geo_proj(obs.geo)
        geometry_types=self.geo_type((~obs.geo_observed).long())
        geo=geometry_features+self.geo_position(obs.geo_position)+geometry_types
        # Context sees measured xyz/uv, never CAD-dependent depth residuals.
        observed_position=obs.geo_position[:,:256].clone();observed_position[:,:,-1]=0
        observed_geo=geometry_features[:,:256]+self.geo_position(observed_position)+geometry_types[:,:256]
        observed_geo_valid=obs.geo_valid[:,:256]
        state=self.state(obs.state)[:,None]
        q=self.core.mask_query+position+prompt[:,None]+dt[:,None]
        obj=self.readout.query[:,:1].expand(b,-1,-1);ctx=self.readout.query[:,1:].expand(b,-1,-1)
        for i,block in enumerate(self.core.blocks):
            q=block(q,current,cv,cb,mem,mv,mb,geo,obs.geo_valid,obs.geo_valid.any(-1).float())
            q=q+self.cad_attn[i](self.cad_norm[i](q),cad,obs.cad_valid)
            patch=self.core.final_norm(q) if i==3 else q
            obj=obj+self.object_attn[i](self.object_norm[i](obj),torch.cat((patch,state),1),
                torch.cat((valid,torch.ones(b,1,device=valid.device,dtype=torch.bool)),1))
            context_source=torch.cat((h+position,observed_geo),1)
            context_valid=torch.cat((valid,observed_geo_valid),1)
            if ctxmem is not None:
                context_source=torch.cat((context_source,ctxmem),1);context_valid=torch.cat((context_valid,cmv),1)
            ctx=ctx+self.context_attn[i](self.context_norm[i](ctx),context_source,context_valid)
        # Final readout uses the same last-layer object query that consumed patches.
        readout,_=self.readout(torch.cat((F.layer_norm(obj,(256,)),F.layer_norm(ctx,(256,))),1),obs.metadata,())
        delta=self.head(readout['latent']).float()
        with torch.autocast(delta.device.type,enabled=False):
            pose=update(obs.base.float(),delta[:,:3],delta[:,3:],obs.diameter.float())
            original=original_pose(pose,obs.center.float())
        next_memory=self.writer(h,position,p,valid,obs,memory)
        surface=self.surface_head(patch).reshape(b,16,16,14,14,5).permute(0,5,1,3,2,4).reshape(b,5,224,224).float()
        return dict(**readout,patch_latent=patch,f_predicted=self.core.feature_last(patch),
            f_mid_predicted=self.core.feature_mid(patch),evidence_logits=logits,
            support_logits=self.core.support(patch).squeeze(-1),
            log_feature_error=self.core.log_error(patch).squeeze(-1).clamp(-14,5),
            pose_centered=pose,pose_original=original,delta_rotvec=delta[:,:3],delta_center_norm=delta[:,3:],
            surface_xyz=surface[:,:3],surface_depth_residual=surface[:,3:4],geometry_valid_logits=surface[:,4:5],
            surface_depth_m=obs.base[:,2,3,None,None,None].float()+surface[:,3:4]*obs.diameter[:,None,None,None].float(),
            read_memory_tokens=0 if mem is None else mem.shape[1]),next_memory


def parameter_category(name):
    if name.startswith('encoder.') or '.encoder.' in name:return None
    if name.startswith('head.'):return 'pose'
    if name.startswith('core.'):return 'predictor'
    return 'new'


def configure_parameters(model):
    for name,p in model.named_parameters():p.requires_grad_(parameter_category(name) is not None)
    model.eval()
