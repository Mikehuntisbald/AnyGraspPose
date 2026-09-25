"""Local frozen CAD descriptors read inside JEPA; no pose or encoder bypass."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .recovered_relation import RecoveredRelationTracker
from .cad_surface_cache import surface_tokens


class SurfaceRead(nn.Module):
    def __init__(self,feature_dim):
        super().__init__()
        self.descriptor=nn.Linear(feature_dim,256)
        self.geometry=nn.Sequential(nn.Linear(21,128),nn.GELU(),nn.Linear(128,256))
        self.query_norm=nn.LayerNorm(256);self.key_norm=nn.LayerNorm(256)
        self.q=nn.Linear(256,128);self.k=nn.Linear(256,128);self.v=nn.Linear(256,256)
        self.out=nn.Linear(256,256);self.gain=nn.Parameter(torch.tensor(-2.))
        nn.init.normal_(self.out.weight,std=.001);nn.init.zeros_(self.out.bias)
        y,x=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
        self.register_buffer('patch_uv',torch.stack(((x.flatten()+.5)/16,(y.flatten()+.5)/16),-1))

    def forward(self,patch,valid,features,geometry,available,observed_xyz=None,observed_valid=None,base=None,confidence=None,query_camera_xyz=None):
        b,n,_=patch.shape;k=features.shape[1]
        # Mask raw inputs as well: dropped CAD cannot leak NaNs or biased values.
        features=torch.where(available[...,None],features,0.)
        geometry=torch.where(available[...,None],geometry,0.)
        tokens=self.key_norm(self.descriptor(features)+self.geometry(geometry))
        q=self.q(self.query_norm(patch)).reshape(b,n,4,32).transpose(1,2)
        keys=self.k(tokens).reshape(b,k,4,32).transpose(1,2)
        values=self.v(tokens).reshape(b,k,4,64).transpose(1,2)
        logits=(q@keys.transpose(-1,-2)).float()/math.sqrt(32)
        if hasattr(self,'rope3d'):
            if observed_xyz is None or observed_valid is None or base is None or confidence is None:
                raise ValueError('CAD RoPE requires explicit measured geometry and confidence')
            logits=self.rope3d(logits,q,keys,observed_xyz,observed_valid,geometry[:,:,:3],available,base,confidence,valid,query_camera_xyz)
        # Soft prior only. No hard image-radius or back-face rejection.
        distance=(self.patch_uv[None,:,None]-geometry[:,None,:,15:17]).square().sum(-1)
        logits=logits-.5*distance[:,None].clamp_max(8.)
        logits=logits.masked_fill(~available[:,None,None],-1e4)
        logits=torch.cat((logits,logits.new_zeros(b,4,n,1)),dim=-1)
        probability=logits.softmax(-1)
        update=(probability[...,:k].to(values.dtype)@values).transpose(1,2).reshape(b,n,256)
        update=self.out(update)*valid[...,None]*available.any(-1)[:,None,None]
        read=patch+self.gain.sigmoid()*update
        log_probability=torch.logsumexp(logits.log_softmax(-1),dim=1)-math.log(4)
        return read,dict(cad_match_log_prob=log_probability,cad_surface_available=available,
                         cad_surface_xyz=geometry[:,:,:3],cad_surface_update_rms=update.detach().float().square().mean().sqrt())


@torch.no_grad()
@torch.autocast('cuda',enabled=False)
def encode_surface(model,scenes,depth,enabled):
    features=[];geometries=[];availability=[]
    for lane,scene in enumerate(scenes):
        bank=surface_tokens(scene.cad,model.cad_surface_cache,model.cad_surface_count)
        xyz=bank['coord'].float();normal=bank['normal'].float();pose=scene.pose.float();d=scene.diameter
        camera=xyz*d@pose[:3,:3].T+pose[:3,3]
        projected=camera@scene.k_crop.float().T;uv=projected[:,:2]/projected[:,2:].clamp_min(.001)
        inside=((uv>=0)&(uv<224)).all(-1)&(camera[:,2]>.001)
        grid=2*(uv+.5)/224-1
        observed=F.grid_sample(depth[lane:lane+1],grid[None,None],mode='nearest',align_corners=False)[0,0,0]
        observed_valid=(observed>0)&torch.isfinite(observed)&inside
        residual=torch.where(observed_valid,(observed-camera[:,2])/d,0.).clamp(-2,2)
        geometry=torch.cat((xyz,normal,bank['color'].float(),camera/d,normal@pose[:3,:3].T,
                            uv/224,residual[:,None],observed_valid[:,None].float(),inside[:,None].float(),
                            ((residual<-.02)&observed_valid)[:,None].float()),-1)
        mask=torch.isfinite(geometry).all(-1)&enabled[lane]&(camera[:,2]>.001)
        features.append(bank['features']);geometries.append(geometry);availability.append(mask)
    return torch.stack(features),torch.stack(geometries),torch.stack(availability)


class CADSurfaceTracker(RecoveredRelationTracker):
    architecture_id='stream_cad_surface_jepa_v12'
    model_version='local-cad-surface-jepa-v12'
    cache_contract='observed-only128-local-cad256-v12'

    def __init__(self,encoder,cached_utonia):
        super().__init__(encoder,cached_utonia)
        self.cad_surface=SurfaceRead(cached_utonia.feature_dim)
        self.cad_surface_count=256

    def extra_frame_inputs(self,obs,memory=None,history_enabled=None):
        if obs.cad_surface_features is None:raise ValueError('Explicit local CAD surface inputs required')
        available=obs.cad_surface_valid&obs.cad_valid.any(-1)[:,None]
        cad_inputs=(obs.cad_surface_features,obs.cad_surface_geometry,available)
        if getattr(self,'cad_rope3d_enabled',False):
            # A no-grad preview must not populate AMP's cached weight casts.
            # These same Linear modules run again with gradients in tensor_frame;
            # reusing detached casts silently freezes their weight/bias gradients.
            device_type=obs.mid.device.type
            with torch.no_grad(),torch.autocast(device_type,
                    enabled=torch.is_autocast_enabled(device_type),
                    dtype=torch.get_autocast_dtype(device_type),cache_enabled=False):
                real=self.core.src_proj(torch.cat((obs.mid,obs.last),-1))
                confidence=self.visibility(real).float().squeeze(-1).sigmoid()
            cad_inputs+=(obs.object_xyz,obs.depth_valid,obs.base,confidence)
        if getattr(self,'staged_rope',None):
            if obs.crop_rays is None or obs.rope_depth_stats is None:raise ValueError('Staged RoPE requires explicit student crop calibration and measured depth statistics')
            cad_inputs+=(obs.crop_rays,obs.rope_depth_stats,obs.diameter)
        return obs.state,(),(obs.object_xyz,obs.depth_valid),cad_inputs

    def prepare_surface_read(self,patch,valid,inputs,levels,layer):
        if not getattr(self,'staged_rope',None) or layer not in (1,3):return inputs,{}
        from .staged_rope import route_surface
        features,geometry,available,observed_xyz,observed_valid,base,confidence,rays,stats,diameter=inputs
        xyz,chosen,trust,_=route_surface(observed_xyz,observed_valid,confidence,stats,base,diameter,rays,valid,self.staged_rope)
        return (features,geometry,available,observed_xyz,chosen,base,trust,xyz),{}

    def refine_surface(self,patch,valid,inputs,levels,before_last,mem,mv,mb):
        if not getattr(self,'staged_rope',None):return patch,levels,{}
        from .staged_rope import route_surface
        features,geometry,available,observed_xyz,observed_valid,base,confidence,rays,stats,diameter=inputs
        # Preserve the trained DPT's four-level input distribution. Decode a
        # complete coarse pass, then recompute only the last shared JEPA block
        # from its original input with recovered-position CAD reading.
        coarse_patch=self.core.final_norm(patch)
        coarse=self.surface_head([*levels[:-1],coarse_patch],valid)
        support=self.core.support(coarse_patch).float().squeeze(-1).sigmoid()
        xyz,chosen,trust,routing=route_surface(observed_xyz,observed_valid,confidence,stats,base,diameter,rays,valid,self.staged_rope,coarse,support)
        updated,metrics=self.cad_surface(before_last,valid,features,geometry,available,observed_xyz,chosen,base,trust,xyz)
        refined=self.core.blocks[-1](updated,valid,mem,mv,mb)
        metrics.update(routing,coarse_surface=coarse,rope_patch_valid=valid)
        return refined,[*levels[:-1],refined],metrics

    def read_surface(self,patch,valid,inputs,layer):
        if layer not in (1,3):return patch,{}
        return self.cad_surface(patch,valid,*inputs)


def migrate_surface(model,source):
    result=model.load_state_dict(source,strict=False)
    allowed=lambda k:k.startswith(('cad_surface.','encoder.'))
    if result.unexpected_keys or any(not allowed(k) for k in result.missing_keys):
        raise ValueError('Unexpected V11 to local-CAD migration')
    model.migration.update(previous_experiment_weights_loaded=True,
        initialization='V11 shared tensors exact; only cad_surface parameters random',new_modules=['cad_surface'])
