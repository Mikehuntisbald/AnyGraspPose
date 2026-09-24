"""Surface orientation from dense XYZ; no independently predicted normal head.

Stencils must belong wholly to one supervised source. Eligibility depends only
on target validity/geometry, never predicted confidence or predicted smoothness.
XYZ is in the same centered object coordinates and diameter units for both.
"""
import math
import torch
from torch.nn import functional as F
from lip.jepa.losses import sample_mean

NORMAL_METRICS=('surface_normal_real','surface_normal_proxy',
                'normal_angle_deg_real','normal_angle_deg_proxy',
                'normal_valid_fraction_real','normal_valid_fraction_proxy')


def normal_field(xyz, stride):
    center=xyz[:,:,stride:-stride,stride:-stride]
    left=xyz[:,:,stride:-stride,:-2*stride];right=xyz[:,:,stride:-stride,2*stride:]
    up=xyz[:,:,:-2*stride,stride:-stride];down=xyz[:,:,2*stride:,stride:-stride]
    dx=F.normalize(right-left,dim=1,eps=1e-4)
    dy=F.normalize(down-up,dim=1,eps=1e-4)
    cross=torch.linalg.cross(dx,dy,dim=1)
    return F.normalize(cross,dim=1,eps=1e-4),cross.norm(dim=1,keepdim=True), (center,left,right,up,down)


def normal_terms(prediction,target,source_mask,stride):
    with torch.autocast(prediction.device.type,enabled=False):
        prediction=prediction.float();target=target.detach().float()
        # Sanitize ineligible targets before differentiation/normalization.
        finite=torch.isfinite(target).all(1,keepdim=True)
        mask=source_mask.bool()&finite
        normal,_,_=normal_field(prediction,stride)
        truth,sine,points=normal_field(target.nan_to_num(),stride)
        valid=mask[:,:,stride:-stride,stride:-stride].clone()
        for m in (mask[:,:,stride:-stride,:-2*stride],mask[:,:,stride:-stride,2*stride:],
                  mask[:,:,:-2*stride,stride:-stride],mask[:,:,2*stride:,stride:-stride]):
            valid=valid&m
        if stride==2:
            for m in (mask[:,:,2:-2,1:-3],mask[:,:,2:-2,3:-1],
                      mask[:,:,1:-3,2:-2],mask[:,:,3:-1,2:-2]):
                valid=valid&m
        center=points[0]
        # Reject discontinuities and degenerate target tangents (diameter units).
        for neighbor in points[1:]:valid=valid&((neighbor-center).norm(dim=1,keepdim=True)<=.05*stride)
        valid=valid&(sine>.05)&((points[2]-points[1]).norm(dim=1,keepdim=True)>1e-4)&((points[4]-points[3]).norm(dim=1,keepdim=True)>1e-4)
        cosine=(normal*truth).sum(1,keepdim=True).clamp(-1,1)
    return 1-cosine, valid, cosine.detach()


def surface_normal_loss(output,target,weights):
    total=output['surface_xyz'].float().sum()*0
    metrics={}
    for name,mask,factor in [('real',target.geometry_real_weight,1.),
                             ('proxy',target.geometry_proxy_weight,weights['cad_feature'])]:
        losses=[];angles=[];fractions=[]
        for stride in (1,2):
            error,valid,cosine=normal_terms(output['surface_xyz'],target.surface_xyz,mask,stride)
            losses.append(sample_mean(error.flatten(1),valid.float().flatten(1))[0])
            angles.append(sample_mean((cosine.acos()*180/math.pi).flatten(1),valid.float().flatten(1))[0])
            fractions.append(valid.float().sum()/mask.float().sum().clamp_min(1))
        loss=torch.stack(losses).mean()
        total=total+factor*loss
        metrics['surface_normal_'+name]=loss.detach()
        metrics['normal_angle_deg_'+name]=torch.stack(angles).mean().detach()
        metrics['normal_valid_fraction_'+name]=torch.stack(fractions).mean().detach()
    return total,metrics


@torch.no_grad()
def normal_diagnostics(output,target,mask):
    errors=[];cosines=[];count=0
    for stride in (1,2):
        error,valid,cosine=normal_terms(output['surface_xyz'],target.surface_xyz,mask,stride)
        errors.append((error*valid).sum());cosines.append((cosine.acos()*180/math.pi*valid).sum())
        count=count+valid.sum()
    if not count:return None
    return dict(stencils=float(count),one_minus_cosine=float(torch.stack(errors).sum()/count),
                angle_deg=float(torch.stack(cosines).sum()/count))
