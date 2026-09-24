"""Full-CAD appearance and paired dense geometry supervision on one patch latent."""
import torch
from torch.nn import functional as F
from lip.jepa.losses import feature_error,normalize,sample_mean

RECONSTRUCTION_METRICS=('real_hidden_feature','cad_proxy_feature','cad_visible_feature','cad_hidden_feature','real_visible_eval',
    'real_hidden_eval','surface_xyz','surface_depth','surface_xyz_real','surface_depth_real','surface_xyz_proxy','surface_depth_proxy',
    'geometry_validity','visibility_support','feature_error')
DEFAULT_WEIGHTS=dict(real_feature=1.,cad_feature=.5,surface_xyz=.5,surface_depth=.5,geometry_validity=.1,visibility_support=.1,feature_error=.05)


def reconstruction_loss(output,target,weights=None):
    weights=DEFAULT_WEIGHTS if weights is None else weights
    def average(value,weight):
        return sample_mean(value.flatten(1),weight.expand_as(value).float().flatten(1))[0]
    def balanced(value,whole,visible,hidden):
        # Cover every valid proxy pixel/patch, with equal per-sample mass for
        # visible and hidden regions; annotation-boundary leftovers remain scored.
        remainder=(whole.float()-visible.float()-hidden.float()).clamp_min(0)
        numerator=value.new_zeros(len(value));denominator=numerator.clone()
        for weight in (visible,hidden,remainder):
            weight=weight.expand_as(value).float();mass=weight.flatten(1).sum(-1)
            numerator+=(value*weight).flatten(1).sum(-1)/mass.clamp_min(1)
            denominator+=(mass>0).float()
        return (numerator/denominator.clamp_min(1)).mean()
    wm,wl=weights.get('feature_mid_weight',.25),weights.get('feature_last_weight',1.)
    pl=feature_error(output['f_predicted'],target.proxy_last);pm=feature_error(output['f_mid_predicted'],target.proxy_mid)
    rl=feature_error(output['f_predicted'],target.real_last);rm=feature_error(output['f_mid_predicted'],target.real_mid)
    proxy=wl*pl+wm*pm
    real=wl*rl+wm*rm
    real_hidden_loss=average(real,target.hidden_real_weight)
    appearance=balanced(proxy,target.proxy_weight,target.proxy_visible_weight,target.proxy_hidden_weight)
    visible_proxy=average(proxy,target.proxy_visible_weight);hidden_proxy=average(proxy,target.proxy_hidden_weight)
    beta=weights.get('geometry_beta',1.)
    xyz=F.smooth_l1_loss(output['surface_xyz'],target.surface_xyz,beta=beta,reduction='none').mean(1,keepdim=True)
    depth=F.smooth_l1_loss(output['surface_depth_residual'],target.surface_depth_residual,beta=beta,reduction='none')
    xyz_loss=balanced(xyz,target.geometry_weight,target.geometry_visible_weight,target.geometry_hidden_weight)
    depth_loss=balanced(depth,target.geometry_weight,target.geometry_visible_weight,target.geometry_hidden_weight)
    known=torch.isfinite(target.geometry_valid_label)
    validity=average(F.binary_cross_entropy_with_logits(output['geometry_valid_logits'],target.geometry_valid_label.nan_to_num(),reduction='none'),known)
    auxiliary=proxy.sum()*0
    for key,label in [('evidence_logits',target.visible_label),('support_logits',target.support_label)]:
        known=torch.isfinite(label)
        auxiliary+=average(F.binary_cross_entropy_with_logits(output[key].float(),label.nan_to_num(),reduction='none'),known)
    proxy_mse=(normalize(output['f_predicted']).detach()-normalize(target.proxy_last)).square().mean(-1)
    real_mse=(normalize(output['f_predicted']).detach()-normalize(target.real_last)).square().mean(-1)
    error=F.smooth_l1_loss(output['log_feature_error'].float(),(proxy_mse+1e-6).log().clamp(-14,5),reduction='none')
    real_error=F.smooth_l1_loss(output['log_feature_error'].float(),(real_mse+1e-6).log().clamp(-14,5),reduction='none')
    uncertainty=average(error,target.proxy_weight)+average(real_error,target.hidden_real_weight)
    # Real and CAD supervision occupy disjoint patch regions. Original visible
    # regions without synthetic occlusion retain independent real diagnostics.
    with torch.no_grad():
        real_visible=average(real.detach(),target.visible_weight);real_hidden=real_hidden_loss.detach()
    total=(weights['real_feature']*real_hidden_loss+weights['cad_feature']*appearance+weights['surface_xyz']*xyz_loss+weights['surface_depth']*depth_loss+
        weights['geometry_validity']*validity+weights['visibility_support']*auxiliary+weights['feature_error']*uncertainty)
    diagnostics=dict(real_hidden_feature=real_hidden_loss.detach(),cad_proxy_feature=appearance.detach(),cad_visible_feature=visible_proxy.detach(),cad_hidden_feature=hidden_proxy.detach(),
        real_visible_eval=real_visible,real_hidden_eval=real_hidden,surface_xyz=xyz_loss.detach(),surface_depth=depth_loss.detach(),
        surface_xyz_real=average(xyz.detach(),target.geometry_real_weight),surface_depth_real=average(depth.detach(),target.geometry_real_weight),
        surface_xyz_proxy=average(xyz.detach(),target.geometry_proxy_weight),surface_depth_proxy=average(depth.detach(),target.geometry_proxy_weight),
        geometry_validity=validity.detach(),visibility_support=auxiliary.detach(),feature_error=uncertainty.detach())

    if weights.get("log_feature_layers",False):
        diagnostics.update(real_feature_mid=average(rm,target.hidden_real_weight).detach(),
            real_feature_last=average(rl,target.hidden_real_weight).detach(),
            proxy_feature_mid=balanced(pm,target.proxy_weight,target.proxy_visible_weight,target.proxy_hidden_weight).detach(),
            proxy_feature_last=balanced(pl,target.proxy_weight,target.proxy_visible_weight,target.proxy_hidden_weight).detach())
    return total,diagnostics
