"""Pose-led supervision without DINO reconstruction or feature-derived targets."""
import torch
from torch.nn import functional as F
from lip.losses import pose_loss
from .recovery_focus import masked_mean,camera_disagreement
from .cad_surface_targets import surface_correspondence_loss

METRICS=('pose','translation','rotation','points','surface_xyz_real','surface_xyz_proxy',
 'surface_depth_real','surface_depth_proxy','camera_consistency_real','camera_consistency_proxy',
 'geometry_validity','visibility_support','cad_surface_real','cad_surface_proxy',
 'coarse_depth_real','coarse_depth_proxy','coarse_validity','rope_measured_fraction',
 'rope_recovered_fraction','rope_fallback_fraction','rope_measured_trust','rope_recovered_trust')
FEATURE_WEIGHTS=('real_feature','cad_feature','feature_error','spatial_centered','spatial_correspondence',
 'local_difference','local_correspondence','surface_normal')


def inactive_feature_parameter(name):
    return name.startswith(('core.feature_mid.','core.feature_last.','core.log_error.'))


def objective(output,target,truth,points,diameter,weights):
    pose,parts=pose_loss(output['pose_centered'],truth,points,diameter)
    stats=dict(pose=pose,**parts);total=weights['pose']*pose
    beta=weights['geometry_beta'];proxy=weights['geometry_proxy_factor']
    xyz=F.smooth_l1_loss(output['surface_xyz'],target.surface_xyz,beta=beta,reduction='none').mean(1,keepdim=True)
    depth=F.smooth_l1_loss(output['surface_depth_residual'],target.surface_depth_residual,beta=beta,reduction='none')
    residual=camera_disagreement(output,target)
    camera=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=beta,reduction='none').mean(1,keepdim=True)
    coarse=output['coarse_surface']
    coarse_depth=F.smooth_l1_loss(coarse[:,3:4],target.surface_depth_residual,beta=beta,reduction='none')
    for name,mask,factor in [('real',target.geometry_real_weight,1.),('proxy',target.geometry_proxy_weight,proxy)]:
        x=masked_mean(xyz,mask);z=masked_mean(depth,mask);cam=masked_mean(camera,mask);cz=masked_mean(coarse_depth,mask)
        stats.update({'surface_xyz_'+name:x,'surface_depth_'+name:z,'camera_consistency_'+name:cam,'coarse_depth_'+name:cz})
        total=total+factor*(weights['surface_xyz']*x+weights['surface_depth']*z+weights['camera_consistency']*cam+
                           weights['coarse_surface']*weights['surface_depth']*cz)
    known=torch.isfinite(target.geometry_valid_label);label=target.geometry_valid_label.nan_to_num()
    validity=masked_mean(F.binary_cross_entropy_with_logits(output['geometry_valid_logits'],label,reduction='none'),known)
    coarse_validity=masked_mean(F.binary_cross_entropy_with_logits(coarse[:,4:5],label,reduction='none'),known)
    support=pose.new_zeros(())
    for key,label in [('evidence_logits',target.visible_label),('support_logits',target.support_label)]:
        support=support+masked_mean(F.binary_cross_entropy_with_logits(output[key].float(),label.nan_to_num(),reduction='none'),torch.isfinite(label))
    cad,cad_metrics=surface_correspondence_loss(output,target)
    total=total+weights['geometry_validity']*(validity+weights['coarse_surface']*coarse_validity)+weights['visibility_support']*support+weights['cad_surface_correspondence']*cad
    stats.update(geometry_validity=validity,visibility_support=support,cad_surface_real=cad_metrics[0],cad_surface_proxy=cad_metrics[1],coarse_validity=coarse_validity)
    eligible=output['rope_patch_valid']&output['cad_surface_available'].any(-1)[:,None]
    for name in ('measured','recovered','fallback'):
        stats['rope_'+name+'_fraction']=(output['rope_'+name+'_mask']&eligible).float().sum()/eligible.float().sum().clamp_min(1)
    for name in ('measured','recovered'):
        mask=output['rope_'+name+'_mask']&eligible
        stats['rope_'+name+'_trust']=(output['rope_'+name+'_trust']*mask).sum()/mask.float().sum().clamp_min(1)
    return total,torch.stack([stats[k].detach() for k in METRICS])
