"""Two-frame replay across the clean/occluded boundary, using only real targets."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from lip.jepa.losses import feature_error, sample_mean


@torch.no_grad()
@torch.autocast('cuda',enabled=False)
def observed_history_support(scenes,truth,visible,target,tolerance_d=.02):
    """Teacher-only audit that a current surface was really measured at anchor.

    GT establishes training correspondence only; it never changes writer inputs,
    memory, attention or student geometry. Labels and depth must both agree.
    """
    from lip.geometry.crop import crop_images
    support=[]
    for lane,(scene,pose,mask) in enumerate(zip(scenes,truth,visible)):
        xyz=target.surface_xyz[lane].float().permute(1,2,0)*scene.diameter
        camera=xyz@pose[:3,:3].float().T+pose[:3,3].float()
        pixel=camera@scene.k_crop.float().T
        uv=pixel[:,:,:2]/pixel[:,:,2:].clamp_min(1e-6)
        grid=2*(uv+.5)/224-1
        observed_mask=crop_images(mask[None].float(),scene.affine,mode='nearest')>.5
        valid=observed_mask&scene.bounds&(scene.depth>0)&torch.isfinite(scene.depth)
        sampled_valid=F.grid_sample(valid.float(),grid[None],mode='nearest',align_corners=False)>.5
        sampled_depth=F.grid_sample(scene.depth.float(),grid[None],mode='nearest',align_corners=False)
        positive=camera[:,:,2][None,None]>.001
        agreement=(sampled_depth-camera[:,:,2][None,None]).abs()<=tolerance_d*scene.diameter
        support.append(sampled_valid&positive&agreement)
    return torch.cat(support)


def write_pair_anchor(model,obs):
    position,_,_,_=model.core.positions(obs.packet,None)
    valid=F.avg_pool2d(obs.packet.pixel_valid.float(),14,14).flatten(1)>=.999
    real=model.core.src_proj(torch.cat((obs.mid,obs.last),-1))
    probability=model.visibility(real).squeeze(-1).sigmoid()*valid
    memory=model.writer(real,position,probability,valid,obs,model.empty_memory(obs))
    # Teach the object route without a context shortcut or target-conditioned write.
    return replace(memory,contexts=())


def without_cad(obs):
    image=torch.cat((obs.geometry_image[:,:2],torch.zeros_like(obs.geometry_image[:,2:])),1)
    return replace(obs,cad_valid=torch.zeros_like(obs.cad_valid),geometry_image=image)


def real_hidden_loss(output,teacher,weights=None):
    def mean(value,weight):
        return sample_mean(value.flatten(1),weight.expand_as(value).float().flatten(1))[0]
    wm=.25 if weights is None else weights.get('feature_mid_weight',.25)
    wl=1. if weights is None else weights.get('feature_last_weight',1.)
    feature=wl*feature_error(output['f_predicted'],teacher.real_last)+wm*feature_error(output['f_mid_predicted'],teacher.real_mid)
    beta=1. if weights is None else weights.get('geometry_beta',1.)
    xyz=F.smooth_l1_loss(output['surface_xyz'],teacher.surface_xyz,beta=beta,reduction='none').mean(1,keepdim=True)
    depth=F.smooth_l1_loss(output['surface_depth_residual'],teacher.surface_depth_residual,beta=beta,reduction='none')
    f=mean(feature,teacher.hidden_real_weight)
    g=.5*mean(xyz,teacher.geometry_real_weight)+.5*mean(depth,teacher.geometry_real_weight)
    focus=f.new_zeros(())
    if weights is not None:
        from .recovery_focus import focus_loss
        focus,_=focus_loss(output,teacher,weights,real_only=True)
    return f+g+focus,torch.stack((f.detach(),g.detach()))


def history_pair_loss(model,previous,current,teacher,weights=None,observed_support=None):
    # Recompute the clean write so TBPTT detach cannot cut this two-frame graph.
    if getattr(model, "trainable_encoder", False):
        # Fresh encoder graph: the clean anchor crosses multiple TBPTT boundaries.
        mid,last=model.encoder(previous.packet.rgb_crop)
        previous=replace(previous,mid=mid,last=last)
    memory=write_pair_anchor(model,previous)
    output,_=model(without_cad(current),memory,torch.ones(len(current.mid),device=current.mid.device,dtype=torch.bool))
    if observed_support is not None:
        real=teacher.geometry_real_weight&observed_support
        patch=F.avg_pool2d(real.float(),14,14).flatten(1)>=.9
        teacher=replace(teacher,hidden_real_weight=teacher.hidden_real_weight*patch,geometry_real_weight=real)
    return real_hidden_loss(output,teacher,weights)
