"""Observed object memory with explicit geometric transport; no learned QK lookup.

Object tokens are selected measured patches, not predicted completions. Context
keeps the independent attention path and is never transported as a rigid object.
"""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from lip.jepa.positions import source_coordinates
from .model import Record, Memory, SpatialWriter, local_pool
from .conv_cross import ConvCrossTracker


class ObservedPatchWriter(SpatialWriter):
    """16 measured representatives per record, stable content plus small residual."""
    def forward(self, h, position, visible, valid, observation, memory):
        source = h + .1 * (self.proj(h) - h + self.ff(h))
        xy = source_coordinates(observation.packet.A_image_to_crop, observation.packet.source_size_wh)
        measured = (valid & observation.depth_valid & torch.isfinite(observation.object_xyz).all(-1)
                    & (observation.object_xyz.norm(dim=-1) <= 1.))
        # Each 4x4 patch cell contributes one actual patch. Features and XYZ use
        # the same index, avoiding an averaged point between different surfaces.
        score = visible.masked_fill(~measured, -1.)
        cell_indices = torch.arange(256, device=h.device).reshape(4,4,4,4).permute(0,2,1,3).reshape(16,16)
        scores = score[:, cell_indices]
        indices = cell_indices[None].expand(len(h),-1,-1).gather(2,scores.argmax(-1,keepdim=True)).squeeze(-1)
        def take(value):
            return value.gather(1,indices[...,None].expand(-1,-1,value.shape[-1]))
        confidence = score.gather(1,indices)
        reliable = confidence >= .2
        quality = (confidence.clamp_min(0)*reliable).sum(-1)/reliable.sum(-1).clamp_min(1)
        obj = Record(take(source),reliable,take(xy),take(observation.object_xyz),reliable,
                     observation.packet.timestamp_s,quality)
        # Context is observed content only; its position is attached by reader.
        features,mass = local_pool(source,valid.float())
        coordinates,_ = local_pool(xy,valid.float())
        xyz,depth_mass = local_pool(observation.object_xyz,(valid&observation.depth_valid).float())
        ctx = Record(features,mass>0,coordinates,xyz,depth_mass>0,observation.packet.timestamp_s,
                     (visible*valid).sum(-1)/valid.sum(-1).clamp_min(1))
        objects = list(memory.objects)
        if len(objects)<4:
            objects.append(obj)
        else:
            def merge(old,new,choose):
                return Record(**{k:torch.where(choose.reshape(-1,*([1]*(getattr(old,k).ndim-1))),
                    getattr(new,k),getattr(old,k)) for k in old.__dataclass_fields__})
            write=obj.valid.any(-1);empty=~objects[0].valid.any(-1)
            objects[0]=merge(objects[0],obj,write&empty)
            quality=torch.stack([r.quality.masked_fill(~r.valid.any(-1),-1) for r in objects[1:]],1)
            index=quality.argmin(1)
            for i in range(1,4):objects[i]=merge(objects[i],obj,write&~empty&(index==i-1))
        return Memory(memory.stream,memory.version,tuple(objects),(memory.contexts+(ctx,))[-4:],observation.packet.timestamp_s)


class GeometricRead(nn.Module):
    """Soft local splatting using estimated pose and intrinsics, with FP32 weights.

    No observed-depth rejection: a foreground occluder must not suppress a
    remembered surface behind it. Render-depth agreement is only a soft prior.
    """
    def __init__(self):
        super().__init__()
        y,x=torch.meshgrid(torch.arange(16)*14+6.5,torch.arange(16)*14+6.5,indexing='ij')
        self.register_buffer('pixels',torch.stack((x,y),-1).reshape(256,2))

    def forward(self, features, xyz, source_valid, ages, quality, base, diameter, state,
                valid, geometry_image, cad_valid):
        with torch.autocast(base.device.type,enabled=False):
            xyz=xyz.float();base=base.float();diameter=diameter.float()
            camera=(xyz*diameter[:,None,None])@base[:,:3,:3].transpose(1,2)+base[:,None,:3,3]
            z=camera[...,2];intrinsics=state[:,10:14].float()*224
            uv=camera[...,:2]/z.clamp_min(1e-5)[...,None]*intrinsics[:,None,:2]+intrinsics[:,None,2:]
            finite=torch.isfinite(camera).all(-1)&torch.isfinite(uv).all(-1)
            active=source_valid&finite&(z>1e-5)&(uv>=0).all(-1)&(uv<224).all(-1)&(ages>0)
            # Sanitize before multiplying: invalid NaNs must not contaminate bmm.
            uv=uv.nan_to_num();z=z.nan_to_num()
            distance=((self.pixels[None,:,None]-uv[:,None])/14).square().sum(-1)
            weights=torch.exp(-distance/(2*1.5**2))*(distance<=3.**2)*active[:,None]
            weights=weights*quality[:,None].float().clamp(0,1)*torch.exp(-ages[:,None].float().clamp_min(0)/2.)
            render_valid=F.avg_pool2d(geometry_image[:,3:4].float(),14).flatten(1)*cad_valid
            render_z=F.avg_pool2d((geometry_image[:,2:3]*geometry_image[:,3:4]).float(),14).flatten(1)
            render_z=render_z/render_valid.clamp_min(1e-6)
            depth_delta=(z[:,None]-base[:,2,3,None,None])/diameter[:,None,None]-render_z[:,:,None]
            agreement=torch.exp(-.5*(depth_delta/.25).square())
            weights=weights*torch.where(render_valid[:,:,None]>.25,agreement,1.)*valid[:,:,None]
            mass=weights.sum(-1);normalized=weights/mass.clamp_min(1e-6)[...,None]
            safe_features=torch.where(active[...,None],features.float(),0.)
            history=normalized@safe_features
            confidence=mass.clamp(0,1)
            age=(normalized*ages[:,None].float()).sum(-1)
            old_depth=((normalized*z[:,None]).sum(-1)-base[:,2,3,None])/diameter[:,None]
            observed_valid=F.avg_pool2d(geometry_image[:,1:2].float(),14).flatten(1)
            observed_z=F.avg_pool2d((geometry_image[:,:1]*geometry_image[:,1:2]).float(),14).flatten(1)/observed_valid.clamp_min(1e-6)
            residual=(old_depth-observed_z).clamp(-2,2)*(observed_valid>0)*(mass>0)
            metadata=torch.stack((confidence,age.clamp_max(10)/10,residual,observed_valid),-1)
        return history.to(features.dtype),metadata


class HistoryFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(256)
        self.adjust=nn.Sequential(nn.Linear(516,256),nn.GELU(),nn.Linear(256,256))
        self.gate=nn.Linear(516,1)
        nn.init.zeros_(self.adjust[-1].weight);nn.init.zeros_(self.adjust[-1].bias)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)

    def forward(self,patch,history,metadata):
        # An immediately usable content path; unsupported history is exactly zero.
        content=self.norm(history)
        incoming=torch.cat((patch,content,metadata.to(patch.dtype)),-1)
        weight=metadata[...,:1].to(patch.dtype)*self.gate(incoming).sigmoid()
        return patch+weight*(content+self.adjust(incoming)),weight


class GeometricHistoryTracker(ConvCrossTracker):
    architecture_id='stream_conv_cross_geohistory_jepa_v10'
    model_version='conv-cross-state-geometric-history-v10'
    cache_contract='observed-selected16-geometric128-v10'

    def __init__(self,encoder,cached_utonia):
        super().__init__(encoder,cached_utonia)
        self.writer=ObservedPatchWriter()
        self.geometric_read=GeometricRead()
        self.history_fusion=HistoryFusion()
        self.weights_version=self.model_version+'/clean-random'
        self.migration.update(history='estimated-pose local splat; context-only QK; observed representative patches')

    def extra_frame_inputs(self,obs,memory=None,history_enabled=None):
        b=len(obs.mid);features=[];xyz=[];valid=[];ages=[];quality=[]
        for i in range(4):
            if i<len(memory.objects):
                r=memory.objects[i];age=(obs.packet.timestamp_s-r.timestamp).float()
                features.append(r.features);xyz.append(r.xyz)
                valid.append(r.valid&r.xyz_valid&history_enabled[:,None]&(age>0)[:,None])
                ages.append(age[:,None].expand(-1,16));quality.append(r.quality[:,None].expand(-1,16))
            else:
                features.append(obs.mid.new_zeros(b,16,256));xyz.append(obs.mid.new_zeros(b,16,3))
                valid.append(obs.cad_valid.new_zeros(b,16));ages.append(obs.mid.new_zeros(b,16));quality.append(obs.mid.new_zeros(b,16))
        return obs.state,tuple(torch.cat(parts,1) for parts in (features,xyz,valid,ages,quality))

    def fuse_history(self,patch,base,diameter,state,valid,geometry_image,cad_valid,history_inputs):
        history,metadata=self.geometric_read(*history_inputs,base,diameter,state,valid,geometry_image,cad_valid)
        patch,weight=self.history_fusion(patch,history,metadata)
        return patch,dict(history_support=metadata[...,0],history_fusion_weight=weight.squeeze(-1))

    def history_sources(self,mem,mv,mb):
        # Object memory is consumed exactly once, geometrically, before JEPA.
        # Context has arbitrary motion and stays on its existing attention path.
        return mem[:,64:],mv[:,64:],mb[:,64:]
