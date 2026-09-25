"""Serial completion losses; GT is excluded from the main student forward.

An explicitly separate oracle rehearsal supervises only the pose readout.
"""
import torch
from torch.nn import functional as F
from lip.losses import pose_loss
from lip.geometry.so3 import log
from .losses import reconstruction_loss
from .recovery_focus import masked_mean, camera_disagreement
from .local_structure import local_structure_loss
from .cad_surface_targets import surface_correspondence_loss
from .staged_rope import coarse_surface_loss


def delta_target(base, truth, diameter):
    return torch.cat((log(truth[:,:3,:3].float() @ base[:,:3,:3].float().transpose(-1,-2)),
        (truth[:,:3,3].float()-base[:,:3,3].float())/diameter[:,None]),-1).detach()


def objective_components(output,target,truth,points,diameter,base,observed_depth,weights,visible_measurement_mask):
    pose,parts=pose_loss(output['pose_centered'],truth,points,diameter)
    recovery,rec=reconstruction_loss(output,target,weights)
    local,_=local_structure_loss(output,target,weights)
    cad,cad_stats=surface_correspondence_loss(output,target)
    coarse,_=coarse_surface_loss(output,target,weights)
    camera=camera_disagreement(output,target)
    camloss=masked_mean(F.smooth_l1_loss(camera,torch.zeros_like(camera),beta=.05,reduction='none').mean(1,keepdim=True),target.geometry_weight)
    # On unoccluded observed pixels only, supervise the CAD identity of the
    # measured point. RGB/depth completion loss remains absent there.
    valid=visible_measurement_mask & (observed_depth>0) & torch.isfinite(observed_depth)
    camera_real=target.camera_rays*observed_depth/diameter[:,None,None,None]-target.camera_translation_d[:,:,None,None]
    true_xyz=torch.einsum('bji,bjhw->bihw',truth[:,:3,:3].float(),camera_real).detach()
    valid=valid & (true_xyz.square().sum(1,keepdim=True)<1.)
    visible_corr=masked_mean(F.smooth_l1_loss(output['surface_xyz'],true_xyz,beta=.05,reduction='none').mean(1,keepdim=True),valid)
    delta=torch.cat((output['delta_rotvec'],output['delta_center_norm']),-1)
    expected=delta_target(base,truth,diameter)
    scale=delta.new_tensor([.174533]*3+[.05]*3)
    direct=F.smooth_l1_loss(delta/scale,expected/scale,beta=.1)
    feature_weights=dict(weights,surface_xyz=0.,surface_depth=0.,geometry_validity=0.,visibility_support=0.)
    feature,_=reconstruction_loss(output,target,feature_weights)
    geometry=recovery-feature+.1*cad+.25*camloss+.5*visible_corr+weights['coarse_surface']*coarse
    appearance=feature+local
    pose_objective=pose+.1*direct
    stats=dict(pose=pose,rotation=parts['rotation'],translation=parts['translation'],
        real_feature=rec['real_hidden_feature'],proxy_feature=rec['cad_proxy_feature'],
        xyz_real=rec['surface_xyz_real'],xyz_proxy=rec['surface_xyz_proxy'],
        depth_real=rec['surface_depth_real'],depth_proxy=rec['surface_depth_proxy'],
        visible_correspondence=visible_corr,camera_consistency=camloss,cad=cad,delta=direct)
    return dict(geometry=geometry,appearance=appearance,pose=pose_objective),{k:v.detach() for k,v in stats.items()}


def objective(*args,**kwargs):
    parts,stats=objective_components(*args,**kwargs)
    return sum(parts.values()),stats


def oracle_readout_loss(model,packet,truth,diameter):
    """Teacher forcing of the small readout only; no JEPA/encoder forward.

The input appearance is cached *student* completion, not teacher features.
Ideal geometry keeps the learned correction map from forgetting its oracle
competence while the real student branch learns to improve its correspondences.
"""
    from lip.geometry.so3 import update
    noise=torch.randn(len(truth),6,device=truth.device)*truth.new_tensor([.15]*3+[.035]*3)
    noise[::4]=0
    base=update(truth,noise[:,:3],noise[:,3:],diameter)
    packet=dict(packet,camera=packet['camera']-noise[:,3:,None,None])
    obj,diagnostics=model.read_completion(packet,base)
    delta=model.head(F.layer_norm(obj[:,0],(256,))).float()*diagnostics['serial_residual_scale'][:,None]
    delta=delta*diagnostics['pose_evidence_available'][:,None]
    expected=delta_target(base,truth,diameter);scale=delta.new_tensor([.174533]*3+[.05]*3)
    return F.smooth_l1_loss(delta/scale,expected/scale,beta=.1)
