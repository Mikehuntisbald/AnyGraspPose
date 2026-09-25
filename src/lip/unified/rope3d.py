"""Optional measured-geometry RoPE inside the unified CAD surface read.

Axial physical XYZ, not video (time,row,column). Frequencies are cycles per
object diameter. A zero-initialized residual gate preserves old checkpoints.
"""
import math
import torch
from torch import nn


def rotate_xyz(value,coordinates,frequencies):
    """value[B,H,N,D], coordinates[B,N,3]; remaining dimensions are untouched."""
    pairs=len(frequencies);width=6*pairs
    if width>value.shape[-1]:raise ValueError('Too many rotary frequencies for head dimension')
    with torch.autocast(value.device.type,enabled=False):
        x=value.float();parts=x[...,:width].reshape(*x.shape[:-1],3,pairs,2)
        angle=coordinates.float()[:,None,:,:,None]*frequencies.float()[None,None,None,None,:]*(2*math.pi)
        cosine,sine=angle.cos(),angle.sin()
        first,second=parts[...,0],parts[...,1]
        rotated=torch.stack((first*cosine-second*sine,first*sine+second*cosine),-1).flatten(-3)
        result=torch.cat((rotated,x[...,width:]),-1)
    return result.to(value.dtype)


class Gated3DRoPE(nn.Module):
    def __init__(self,heads=4,frequencies=(.5,1.,2.,4.,8.)):
        super().__init__()
        if len(frequencies)!=5 or any(not math.isfinite(f) or f<=0 for f in frequencies):
            raise ValueError('CAD RoPE expects five positive finite frequencies')
        self.register_buffer('frequencies',torch.tensor(frequencies,dtype=torch.float32),persistent=False)
        self.gain=nn.Parameter(torch.zeros(heads))

    def forward(self,logits,q,k,observed_xyz,observed_valid,cad_xyz,cad_available,base,confidence,patch_valid,query_camera_xyz=None):
        # Observation XYZ was inverse-transformed by the estimated base pose.
        # Rotate both streams into current camera axes, centered at base t / d.
        # q=(X_observed-t_base)/d; k=R_base X_CAD/d. Their difference retains
        # the current estimated translation/rotation error. Never use GT here.
        with torch.autocast(q.device.type,enabled=False):
            coordinate=observed_xyz if query_camera_xyz is None else query_camera_xyz
            qvalid=observed_valid & patch_valid & torch.isfinite(coordinate).all(-1)
            kvalid=cad_available & torch.isfinite(cad_xyz).all(-1)
            rotation=base[:,:3,:3].detach().float().transpose(-1,-2)
            query_xyz=torch.where(qvalid[...,None],coordinate.detach().float(),0.)
            if query_camera_xyz is None:query_xyz=query_xyz@rotation
            key_xyz=torch.where(kvalid[...,None],cad_xyz.detach().float(),0.)@rotation
            # Confidence only weights this positional residual. It cannot learn
            # to escape the RoPE loss by suppressing visibility through this edge.
            trust=confidence.detach().float().nan_to_num().clamp(0,1)*qvalid
        rq=rotate_xyz(q,query_xyz,self.frequencies);rk=rotate_xyz(k,key_xyz,self.frequencies)
        rotated=(rq@rk.transpose(-1,-2)).float()/math.sqrt(q.shape[-1])
        weight=self.gain.tanh()[None,:,None,None]*trust[:,None,:,None]*kvalid[:,None,None,:]
        return logits+weight*(rotated-logits)


def enable_cad_rope3d(model,config=None):
    """Attach four zero-initialized scalars after loading, or via build config."""
    if model.architecture_id not in ('stream_cad_surface_jepa_v12','stream_serial_completion_jepa_v21'):
        raise ValueError('CAD RoPE candidate requires the existing local surface reader')
    if hasattr(model.cad_surface,'rope3d'):raise ValueError('CAD RoPE already enabled')
    config={} if config is None else config
    model.cad_surface.rope3d=Gated3DRoPE(frequencies=tuple(config.get('cycles_per_diameter',(.5,1,2,4,8)))).to(model.query.device)
    model.cad_rope3d_enabled=True
    model.model_version=getattr(model,'model_version','local-cad')+'-rope3d-candidate'
    model.weights_version=model.model_version+'/candidate'


def allowed_migration_keys(config):
    return {'cad_surface.rope3d.gain'} if config.get('cad_rope3d',{}).get('enabled',False) else set()
