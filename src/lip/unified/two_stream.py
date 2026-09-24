"""Clean initialization: two dense input streams, one fusion, one JEPA backbone."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from lip.jepa.predictor import SafeAttention
from lip.jepa.positions import PositionEncoding
from lip.jepa.contracts import FramePacket
from lip.data.jepa_pairs import normalize_rgb
from lip.engine.stream_state import FrameMeta
from lip.geometry.so3 import update, original_pose
from .model import Memory, SpatialWriter
from .features import Observation, camera_points, usable_depth


class JointBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(256) for _ in range(3)])
        self.spatial = SafeAttention()
        self.history = SafeAttention()
        self.ff = nn.Sequential(nn.Linear(256,1024),nn.GELU(),nn.Linear(1024,256))

    def forward(self, x, valid, memory, memory_valid, memory_bias):
        h = self.norms[0](x)
        x = x + self.spatial(h,h,valid)
        if not getattr(self, "disable_history", False):
            x = x + self.history(self.norms[1](x),memory,memory_valid,memory_bias)
        return x + self.ff(self.norms[2](x))


class TwoStreamTracker(nn.Module):
    architecture_id = 'stream_two_input_jepa_v9'
    model_version = 'two-input-clean-v9'
    cache_contract = 'observed-local128-two-input-v9'
    online_geometry = 'joint_geometry_image'

    def __init__(self, encoder, cached_utonia):
        super().__init__()
        self.encoder=encoder;self.utonia=cached_utonia;self.weights_version='clean-random'
        self.core=nn.Module()
        self.core.src_proj=nn.Linear(768,256)
        self.core.positions=PositionEncoding(256)
        self.core.blocks=nn.ModuleList([JointBlock() for _ in range(4)])
        self.core.final_norm=nn.LayerNorm(256)
        self.core.feature_mid=nn.Linear(256,384);self.core.feature_last=nn.Linear(256,384)
        self.core.support=nn.Linear(256,1);self.core.log_error=nn.Linear(256,1)
        self.core.memory_types=nn.Embedding(2,256)
        self.core.memory_age_rate=nn.Parameter(torch.tensor(-2.))
        self.appearance=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU())
        layers=[];channels=9
        for width in (32,64,128):
            layers += [nn.Conv2d(channels,width,3,stride=2,padding=1),nn.GroupNorm(8,width),nn.GELU(),
                       nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.GELU()]
            channels=width
        self.geometry=nn.Sequential(*layers,nn.Conv2d(128,256,1))
        # Separable area averaging avoids CUDA adaptive-pool's nondeterministic
        # backward for the 28x28 CNN grid -> 16x16 DINO grid conversion.
        pool=torch.zeros(16,28)
        for i in range(16):
            start=i*28//16;end=((i+1)*28+15)//16
            pool[i,start:end]=1/(end-start)
        self.register_buffer('geometry_pool',pool)
        self.cad_prior=nn.Linear(cached_utonia.feature_dim,256)
        self.fusion=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU(),nn.Linear(256,256))
        self.query=nn.Parameter(torch.randn(1,1,256)*.02)
        self.object_attn=SafeAttention();self.object_norm=nn.LayerNorm(256)
        self.head=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,6))
        # Random, small initial updates keep an untrained closed loop well-defined.
        nn.init.normal_(self.head[-1].weight,std=1e-4);nn.init.zeros_(self.head[-1].bias)
        self.surface_head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,14*14*5))
        self.visibility=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,1))
        self.writer=SpatialWriter()
        nn.init.xavier_uniform_(self.writer.proj.weight)
        nn.init.xavier_uniform_(self.writer.ff[-1].weight)
        self.memory_position=nn.Linear(6,256)
        self.compiled_frame=None
        self.migration=dict(kind='clean_random',source_step=0,source_sampler_position=0,
                            previous_experiment_weights_loaded=False,optimizer_reset=True,memory_reset=True)

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();self.utonia.eval();return self

    def empty_memory(self,obs):return Memory(obs.packet.stream_id,self.weights_version)

    def enable_compilation(self,enabled=True):
        self.compiled_frame=torch.compile(self.tensor_frame,fullgraph=True,dynamic=False) if enabled else None

    def tensor_frame(self,mid,last,cad_mid,cad_last,cad_valid,geometry_image,static_cad,
                     base,diameter,center,position,prompt,dt,valid,mem,mv,mb):
        b=len(mid)
        real=self.core.src_proj(torch.cat((mid,last),-1))
        cad=self.core.src_proj(torch.cat((cad_mid,cad_last),-1))*cad_valid[...,None]
        appearance=self.appearance(torch.cat((real,cad),-1))
        spatial=self.geometry(geometry_image)
        geometry=(self.geometry_pool@spatial@self.geometry_pool.T).flatten(2).transpose(1,2)
        geometry=geometry+self.cad_prior(static_cad)[:,None]*cad_valid.any(-1)[:,None,None]
        patch=self.fusion(torch.cat((appearance,geometry),-1))+position+prompt[:,None]+dt[:,None]
        for block in self.core.blocks:patch=block(patch,valid,mem,mv,mb)
        patch=self.core.final_norm(patch)
        obj=self.query.expand(b,-1,-1)
        obj=obj+self.object_attn(self.object_norm(obj),patch,valid)
        latent=F.layer_norm(obj[:,0],(256,))
        delta=self.head(latent).float()
        with torch.autocast(delta.device.type,enabled=False):
            pose=update(base.float(),delta[:,:3],delta[:,3:],diameter.float())
            original=original_pose(pose,center.float())
        surface=self.surface_head(patch).reshape(b,16,16,14,14,5).permute(0,5,1,3,2,4).reshape(b,5,224,224).float()
        logits=self.visibility(real).squeeze(-1)
        return dict(latent=latent,latent_object=latent,latent_context=real.mean(1),patch_latent=patch,
            pose_centered=pose,pose_original=original,delta_rotvec=delta[:,:3],delta_center_norm=delta[:,3:],
            f_predicted=self.core.feature_last(patch),f_mid_predicted=self.core.feature_mid(patch),
            evidence_logits=logits,support_logits=self.core.support(patch).squeeze(-1),
            log_feature_error=self.core.log_error(patch).squeeze(-1).clamp(-14,5),
            surface_xyz=surface[:,:3],surface_depth_residual=surface[:,3:4],geometry_valid_logits=surface[:,4:5],
            surface_depth_m=base[:,2,3,None,None,None].float()+surface[:,3:4]*diameter[:,None,None,None].float(),
            writer_h=real,writer_p=logits.sigmoid()*valid)

    def extra_frame_inputs(self, obs, memory=None, history_enabled=None):
        """Explicit subclass inputs, also passed through the compiled frame."""
        return ()

    def forward(self,obs,memory=None,history_enabled=None):
        packet=obs.packet;b=len(obs.mid)
        if not getattr(self,'trusted_training_inputs',False):packet.validate()
        history_disabled=getattr(self,"disable_history",False)
        memory=self.empty_memory(obs) if memory is None or history_disabled else memory
        if memory.stream!=packet.stream_id or memory.version!=self.weights_version:raise ValueError('Stale/cross-stream memory')
        if not getattr(self,'trusted_training_inputs',False) and memory.timestamp is not None and not torch.all(packet.timestamp_s>memory.timestamp):raise ValueError('Noncausal memory')
        enabled=torch.ones(b,device=obs.mid.device,dtype=torch.bool) if history_enabled is None else history_enabled
        position,prompt,dt,_=self.core.positions(packet,memory.timestamp)
        valid=F.avg_pool2d(packet.pixel_valid.float(),14,14).flatten(1)>=.999
        features=[];masks=[];bias=[]
        for kind,records in enumerate(() if history_disabled else (memory.objects,memory.contexts)):
            for i in range(4):
                if i<len(records):
                    r=records[i];spatial=torch.cat((r.xyz*r.xyz_valid[...,None],r.xy,r.xyz_valid[...,None].float()),-1)
                    features.append(r.features+self.core.memory_types.weight[kind]+self.memory_position(spatial))
                    age=packet.timestamp_s-r.timestamp
                    masks.append(r.valid&(age>0)[:,None]&enabled[:,None])
                    bias.append(-F.softplus(self.core.memory_age_rate)*age.float().log1p()[:,None].expand(-1,16))
                else:
                    features.append(self.core.memory_types.weight[kind][None,None].expand(b,16,-1)*0)
                    masks.append(valid.new_zeros(b,16));bias.append(self.core.memory_age_rate.float().expand(b,16)*0)
        if history_disabled:
            features=[obs.mid.new_zeros(b,0,256)];masks=[valid[:,:0]];bias=[obs.mid.new_zeros(b,0)]
        result=(self.compiled_frame or self.tensor_frame)(obs.mid,obs.last,obs.cad_mid,obs.cad_last,obs.cad_valid,
            obs.geometry_image,obs.geo,obs.base,obs.diameter,obs.center,position,prompt,dt,valid,
            torch.cat(features,1),torch.cat(masks,1),torch.cat(bias,1),*self.extra_frame_inputs(obs,memory,enabled))
        writer_h=result.pop('writer_h');writer_p=result.pop('writer_p')
        next_memory=self.empty_memory(obs) if history_disabled else self.writer(writer_h,position,writer_p,valid,obs,memory)
        result['read_memory_tokens']=sum(len(rs)*16 for rs in (memory.objects,memory.contexts))
        return result,next_memory


@torch.autocast('cuda',enabled=False)
def encode_two_stream(model,scenes,masks=None,cad_enabled=None,frame_id=0,occlusions=None):
    if masks is not None and (occlusions is not None or any(bool(m.any()) for m in masks)):
        raise ValueError('Artificial occlusion requires textured RGB-D donors')
    from lip.geometry.crop import geometry_channels
    b=len(scenes);device=scenes[0].rgb.device
    occlusions=[None]*b if occlusions is None else occlusions
    cad_enabled=torch.ones(b,device=device,dtype=torch.bool) if cad_enabled is None else cad_enabled
    rgb=torch.cat([s.rgb if o is None else o.rgb for s,o in zip(scenes,occlusions)])
    depth=usable_depth(torch.cat([s.depth if o is None else o.depth for s,o in zip(scenes,occlusions)]))
    bounds=torch.cat([s.bounds for s in scenes]);depth=torch.where(bounds,depth,0.)
    base=torch.stack([s.pose for s in scenes]);diameter=rgb.new_tensor([s.diameter for s in scenes])
    render_rgb=torch.stack([s.render['rgb'] for s in scenes])
    timing=getattr(model,'profile_timing',None);scope=timing.record('dino_observation_and_reference') if timing is not None else nullcontext()
    with scope,torch.autocast(device.type,dtype=torch.bfloat16):mid,last=model.encoder(normalize_rgb(torch.cat((rgb,render_rgb))))
    geometry=[];xyz=[];depth_valid=[];cad=[]
    for i,s in enumerate(scenes):
        rd=torch.where(bounds[i],s.render['depth'],0.)
        g=geometry_channels(depth[i:i+1],rd[None],s.render['xyz'][None],diameter[i],base[i,2,3])
        # Keep only measured depth/validity when the CAD reference is dropped.
        g[:,2:]*=cad_enabled[i];geometry.append(g)
        camera=camera_points(depth[i:i+1],s.k_crop)
        local=(camera-base[i,:3,3])@base[i,:3,:3]/diameter[i]
        valid=depth[i:i+1]>0;mass=F.avg_pool2d(valid.float(),14,14)
        pooled=F.avg_pool2d(local.permute(2,0,1)[None]*valid,14,14)/mass.clamp_min(1e-6)
        xyz.append(pooled.flatten(2).transpose(1,2)[0]);depth_valid.append(mass.flatten()>0)
        cad.append(s.cad['features'].mean(0))
    packet=FramePacket(normalize_rgb(rgb),bounds,torch.stack([s.affine for s in scenes]),
        torch.tensor([s.timestamp for s in scenes],device=device,dtype=torch.float64),tuple(s.stream for s in scenes),
        rgb.new_tensor([[32.,32.,192.,192.]]).expand(b,-1),rgb.new_tensor([s.size_wh for s in scenes]))
    pixel_valid=F.avg_pool2d(bounds.float(),14,14).flatten(1)>=.999
    metadata=FrameMeta(packet.timestamp_s,torch.full((b,),frame_id,device=device,dtype=torch.long),
        torch.arange(b,device=device),torch.cat((F.adaptive_max_pool2d(bounds.float(),4).flatten(1)>0,pixel_valid.any(-1,keepdim=True)),1),torch.zeros(b,17,device=device))
    empty=rgb.new_empty(b,0,6);empty_mask=pixel_valid[:,:0]
    local_cad=(None,None,None)
    if hasattr(model,'cad_surface'):
        from .cad_surface import encode_surface
        local_cad=encode_surface(model,scenes,depth,cad_enabled)
    return Observation(packet,mid[:b],last[:b],mid[b:],last[b:],pixel_valid&cad_enabled[:,None],
        torch.stack(cad),empty,empty_mask,empty_mask,torch.stack(xyz),torch.stack(depth_valid),
        torch.stack([s.state for s in scenes]),base,diameter,torch.stack([s.center for s in scenes]),metadata,
        geometry_image=torch.cat(geometry),cad_surface_features=local_cad[0],cad_surface_geometry=local_cad[1],cad_surface_valid=local_cad[2])
