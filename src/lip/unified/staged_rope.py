"""Measured-first, recovered-later camera geometry for CAD RoPE.

Only student RGB-D, crop calibration and intermediate JEPA predictions enter
the routing function. Ground-truth tensors are confined to coarse_surface_loss.
"""
import torch
from torch.nn import functional as F
from .recovery_focus import masked_mean,camera_disagreement

STAGED_METRICS=('coarse_xyz_real','coarse_xyz_proxy','coarse_depth_real','coarse_depth_proxy',
                'coarse_camera_real','coarse_camera_proxy','coarse_validity',
                'rope_measured_fraction','rope_recovered_fraction','rope_fallback_fraction',
                'rope_measured_trust','rope_recovered_trust')


def enable_staged_rope(model,plan):
    if not getattr(model,'cad_rope3d_enabled',False) or getattr(model,'surface_decoder_kind','')!='dpt':
        raise ValueError('Staged RoPE requires the existing DPT and CAD RoPE modules')
    model.staged_rope=dict(visible_threshold=float(plan.get('visible_threshold',.7)),
        measured_min_mass=float(plan.get('measured_min_mass',.5)),
        measured_max_std_d=float(plan.get('measured_max_std_d',.05)),
        recovered_threshold=float(plan.get('recovered_threshold',.5)),
        recovered_support_threshold=float(plan.get('recovered_support_threshold',.7)),
        recovered_max_trust=float(plan.get('recovered_max_trust',.5)))
    for key in ('visible_threshold','measured_min_mass','recovered_threshold','recovered_support_threshold','recovered_max_trust'):
        if not 0<model.staged_rope[key]<=1:raise ValueError('Invalid staged RoPE confidence threshold')
    if model.staged_rope['measured_max_std_d']<=0:raise ValueError('Invalid measured depth spread threshold')
    model.model_version='local-cad-measured-first-recovered-later-v18'


@torch.no_grad()
def measured_depth_stats(depth,base,diameter):
    with torch.autocast(depth.device.type,enabled=False):
        valid=torch.isfinite(depth)&(depth>0)
        value=torch.where(valid,(depth.float()-base[:,2,3,None,None,None])/diameter[:,None,None,None],0.)
        mass=F.avg_pool2d(valid.float(),14,14)
        mean=F.avg_pool2d(value,14,14)/mass.clamp_min(1e-6)
        variance=(F.avg_pool2d(value.square(),14,14)/mass.clamp_min(1e-6)-mean.square()).clamp_min(0)
    return torch.stack((mass.flatten(1),variance.sqrt().flatten(1)),-1)


@torch.no_grad()
def route_surface(observed_xyz,observed_valid,visibility,depth_stats,base,diameter,rays,valid,plan,coarse=None,object_support=None):
    """Disjoint source selection; coordinates are camera XYZ centered at base t/d."""
    with torch.autocast(observed_xyz.device.type,enabled=False):
        visibility=visibility.float().nan_to_num().clamp(0,1)
        finite=torch.isfinite(observed_xyz).all(-1)
        measured=(valid&observed_valid&finite&(visibility>=plan['visible_threshold'])&
                  (depth_stats[...,0]>=plan['measured_min_mass'])&(depth_stats[...,1]<=plan['measured_max_std_d']))
        measured_xyz=torch.where(finite[...,None],observed_xyz.detach().float(),0.)@base[:,:3,:3].float().transpose(-1,-2)
        measured_trust=visibility*measured
        recovered=torch.zeros_like(measured);recovered_xyz=torch.zeros_like(measured_xyz)
        recovered_trust=torch.zeros_like(measured_trust)
        if coarse is not None:
            if object_support is None:raise ValueError('Recovered positions require explicit predicted object support')
            # Canonical XYZ rotated by the estimated base would re-impose its
            # pose error. Lift predicted camera depth through actual crop rays.
            depth=base[:,2,3,None,None,None].float()+coarse[:,3:4].detach().float()*diameter[:,None,None,None]
            probability=coarse[:,4:5].detach().float().sigmoid().nan_to_num()
            positive=torch.isfinite(depth)&(depth>.001)&(depth<65.5345)&torch.isfinite(rays).all(1,keepdim=True)
            weight=probability*positive
            mass=F.avg_pool2d(weight,14,14)
            camera=torch.where(positive,depth,0.)*rays.float().nan_to_num()
            # Center before pooling to avoid subtracting two large camera-Z
            # means when the quantity of interest is a small pose residual.
            centered=(camera-base[:,:3,3,None,None])/diameter[:,None,None,None]
            mean=F.avg_pool2d(centered*weight,14,14)/mass.clamp_min(1e-6)
            recovered_xyz=mean.flatten(2).transpose(1,2)
            confidence=mass.flatten(1)
            support=object_support.detach().float().nan_to_num().clamp(0,1)
            recovered=valid&~measured&(confidence>=plan['recovered_threshold'])&(support>=plan['recovered_support_threshold'])&torch.isfinite(recovered_xyz).all(-1)
            # A visible RGB patch can still lack trustworthy sensor depth.
            # Priority is enforced by disjoint masks; do not suppress recovery
            # merely because RGB visibility is high when measured geometry fails.
            recovered_trust=plan['recovered_max_trust']*confidence*support*recovered
        positions=torch.where(measured[...,None],measured_xyz,torch.where(recovered[...,None],recovered_xyz,0.))
        chosen=measured|recovered
        trust=measured_trust+recovered_trust
    return positions,chosen,trust,dict(rope_measured_mask=measured,rope_recovered_mask=recovered,
        rope_fallback_mask=valid&~chosen,rope_measured_trust=measured_trust,rope_recovered_trust=recovered_trust)


def coarse_surface_loss(output,target,weights):
    """Direct geometry supervision for the shared decoder's intermediate pass."""
    coarse=output['coarse_surface']
    xyz=F.smooth_l1_loss(coarse[:,:3],target.surface_xyz,beta=weights.get('geometry_beta',.05),reduction='none').mean(1,keepdim=True)
    depth=F.smooth_l1_loss(coarse[:,3:4],target.surface_depth_residual,beta=weights.get('geometry_beta',.05),reduction='none')
    residual=camera_disagreement(dict(surface_xyz=coarse[:,:3],surface_depth_residual=coarse[:,3:4]),target)
    camera=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=.05,reduction='none').mean(1,keepdim=True)
    total=coarse.sum()*0;parts={}
    for name,mask,factor in [('real',target.geometry_real_weight,1.),('proxy',target.geometry_proxy_weight,weights['cad_feature'])]:
        x=masked_mean(xyz,mask);z=masked_mean(depth,mask);c=masked_mean(camera,mask)
        parts['coarse_xyz_'+name]=x.detach();parts['coarse_depth_'+name]=z.detach();parts['coarse_camera_'+name]=c.detach()
        total=total+factor*(weights['surface_xyz']*x+weights['surface_depth']*z+weights.get('camera_consistency',0)*c)
    known=torch.isfinite(target.geometry_valid_label)
    validity=masked_mean(F.binary_cross_entropy_with_logits(coarse[:,4:5],target.geometry_valid_label.nan_to_num(),reduction='none'),known)
    total=total+weights['geometry_validity']*validity;parts['coarse_validity']=validity.detach()
    eligible=output['rope_patch_valid']&output['cad_surface_available'].any(-1)[:,None]
    denominator=eligible.float().sum().clamp_min(1)
    for name in ('measured','recovered','fallback'):
        parts['rope_'+name+'_fraction']=(output['rope_'+name+'_mask']&eligible).float().sum()/denominator
    for name in ('measured','recovered'):
        mask=output['rope_'+name+'_mask']&eligible
        parts['rope_'+name+'_trust']=(output['rope_'+name+'_trust']*mask).sum()/mask.float().sum().clamp_min(1)
    return total,{k:v.detach() for k,v in parts.items()}


def preserve_adam(optimizer,source):
    from .horizon_resume import exact
    expected=optimizer.state_dict()['param_groups'];saved=source['optimizer']
    if [g['names'] for g in expected]!=[g['names'] for g in saved['param_groups']]:
        raise ValueError('Staged RoPE continuation changed optimizer parameter order')
    optimizer.load_state_dict(saved)
    if not exact(optimizer.state_dict(),saved):raise ValueError('Adam state was not restored exactly')
