"""Serial completion supervision; GT never enters student forward/readout."""
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


def objective(output,target,truth,points,diameter,base,observed_depth,weights):
    pose,parts=pose_loss(output['pose_centered'],truth,points,diameter)
    recovery,rec=reconstruction_loss(output,target,weights)
    local,_=local_structure_loss(output,target,weights)
    cad,cad_stats=surface_correspondence_loss(output,target)
    coarse,_=coarse_surface_loss(output,target,weights)
    camera=camera_disagreement(output,target)
    camloss=masked_mean(F.smooth_l1_loss(camera,torch.zeros_like(camera),beta=.05,reduction='none').mean(1,keepdim=True),target.geometry_weight)
    # On unoccluded observed pixels only, supervise the CAD identity of the
    # measured point. RGB/depth completion loss remains absent there.
    visible=target.visible_label.nan_to_num().reshape(-1,1,16,16).repeat_interleave(14,-2).repeat_interleave(14,-1)>.9
    valid=visible & (observed_depth>0) & torch.isfinite(observed_depth)
    camera_real=target.camera_rays*observed_depth/diameter[:,None,None,None]-target.camera_translation_d[:,:,None,None]
    true_xyz=torch.einsum('bji,bjhw->bihw',truth[:,:3,:3].float(),camera_real).detach()
    valid=valid & (true_xyz.square().sum(1,keepdim=True)<1.)
    visible_corr=masked_mean(F.smooth_l1_loss(output['surface_xyz'],true_xyz,beta=.05,reduction='none').mean(1,keepdim=True),valid)
    delta=torch.cat((output['delta_rotvec'],output['delta_center_norm']),-1)
    expected=delta_target(base,truth,diameter)
    scale=delta.new_tensor([.174533]*3+[.05]*3)
    direct=F.smooth_l1_loss(delta/scale,expected/scale,beta=.1)
    total=pose+recovery+local+.1*cad+.25*camloss+.5*visible_corr+.1*direct+weights['coarse_surface']*coarse
    stats=dict(pose=pose,rotation=parts['rotation'],translation=parts['translation'],
        real_feature=rec['real_hidden_feature'],proxy_feature=rec['cad_proxy_feature'],
        xyz_real=rec['surface_xyz_real'],xyz_proxy=rec['surface_xyz_proxy'],
        depth_real=rec['surface_depth_real'],depth_proxy=rec['surface_depth_proxy'],
        visible_correspondence=visible_corr,camera_consistency=camloss,cad=cad,delta=direct)
    return total,{k:v.detach() for k,v in stats.items()}
