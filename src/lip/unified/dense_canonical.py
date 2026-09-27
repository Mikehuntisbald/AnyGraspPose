"""JEPA-conditioned dense CAD correspondence without patchwise XYZ quantization.

Only canonical XYZ is read from the estimated-view CAD raster. Camera depth
and validity remain the separately predicted fields. No teacher enters forward.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .cad_transport import pixel_grid, sample_reference, transport_targets


class DenseCanonicalReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Conv2d(25,32,3,padding=1),nn.GELU(),
                                nn.Conv2d(32,3,3,padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():self.head[-1].bias[2]=1.

    def forward(self,dense,fallback,geometry,cad_valid):
        b,_,h,w=fallback.shape
        bounds=cad_valid.reshape(b,1,16,16).repeat_interleave(14,-2).repeat_interleave(14,-1)
        available=(geometry[:,3:4]>0)&bounds&torch.isfinite(geometry[:,2:7]).all(1,keepdim=True)
        reference=torch.cat((geometry[:,4:7],geometry[:,2:3]),1).detach().float()
        reference=torch.where(available,reference,0.)
        difference=torch.where(available,fallback[:,:4].float()-reference,0.)
        condition=torch.cat((reference,available.float(),difference),1)
        raw=self.head(torch.cat((dense,condition.to(dense.dtype)),1)).float()
        with torch.autocast(raw.device.type,enabled=False):
            flow=112*raw[:,:2].tanh()
            uv=pixel_grid(b,h,w,raw.device)+flow
            warped,mass=sample_reference(reference[:,:3],available,uv)
            soft_gate=raw[:,2:3].sigmoid()*(mass>=.999).detach()
            # Forward chooses one source, avoiding a fictitious blend of two
            # surfaces; backward supplies a soft gate surrogate for learning.
            gate=soft_gate+((soft_gate>.5).float()-soft_gate).detach()
            blend=fallback[:,:3].float()+gate*(warped-fallback[:,:3].float())
            hard_xyz=torch.where(soft_gate>.5,warped,fallback[:,:3].float())
            xyz=hard_xyz.detach()+(blend-blend.detach())
            surface=torch.cat((xyz,fallback[:,3:].float()),1)
        return surface,dict(flow=flow,uv=uv,warped_xyz=warped,mass=mass,
                            gate_logits=raw[:,2:3],gate=gate,soft_gate=soft_gate,
                            fallback_xyz=fallback[:,:3])


@torch.no_grad()
def read_preference(result,truth,margin=.005):
    """Teacher-only branch preference, not a claim of absolute confidence."""
    if margin<0:raise ValueError('Read preference margin must be nonnegative')
    read_error=(result['warped_xyz'].float()-truth.float()).norm(dim=1,keepdim=True)
    fallback_error=(result['fallback_xyz'].float()-truth.float()).norm(dim=1,keepdim=True)
    return (read_error+margin<fallback_error)&(result['mass']>=.999)


@torch.autocast('cuda',enabled=False)
def dense_canonical_loss(rounds,target,observation,crop_k,visible,gate_target='supported',preference_margin=.005):
    """Independent dense correspondence labels; no predicted-gate masking."""
    truth=transport_targets(target.cad_geometry_xyz,observation.geometry_image,
                            observation.base,observation.diameter,crop_k,observation.cad_valid)
    masks=[('real',target.geometry_real_weight,1.),('proxy',target.geometry_proxy_weight,.5),
           ('observed',visible,.5)]
    def average(value,mask):
        count=mask.flatten(1).sum(-1)
        per=(value*mask).flatten(1).sum(-1)/count.clamp_min(1)
        return (per*(count>0)).sum()/(count>0).sum().clamp_min(1)
    total=rounds[0]['flow'].sum()*0;metrics={}
    for stage,result in enumerate(rounds):
        value=total*0
        error=F.smooth_l1_loss(result['flow']/224.,truth['flow']/224.,beta=1/224.,reduction='none').mean(1,keepdim=True)
        xyz=F.smooth_l1_loss(result['warped_xyz'],target.cad_geometry_xyz,beta=.02,reduction='none').mean(1,keepdim=True)
        if gate_target=='supported':gate_label=truth['supported']
        elif gate_target=='better_than_fallback':gate_label=read_preference(result,target.cad_geometry_xyz,preference_margin)
        else:raise ValueError('Unknown dense canonical gate target: '+gate_target)
        gate=F.binary_cross_entropy_with_logits(result['gate_logits'],gate_label.float(),reduction='none')
        for name,mask,factor in masks:
            domain=mask&target.cad_geometry_valid
            supported=domain&truth['supported']
            value=value+factor*(12.5*average(error,supported)+.5*average(xyz,supported)+.1*average(gate,domain))
            metrics[f'dense{stage}_{name}_epe']=average((result['flow'].detach()-truth['flow']).norm(dim=1,keepdim=True),supported)
            metrics[f'dense{stage}_{name}_support']=average(truth['supported'].float(),domain)
        total=total+(1. if stage==len(rounds)-1 else .5)*value
    return total,metrics
