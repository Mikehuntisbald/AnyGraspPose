"""Keep metric geometry the primary objective of the restoration backbone.

The forward pass is unchanged. During the secondary backward, only restoration
parameters are scaled; pose and decoded-appearance readout parameters keep their
full gradients. The scale is measured at the shared final patch, not inferred
from incomparable scalar loss values. This is a gradient-budget intervention,
not a guarantee about finite Adam updates or held-out accuracy.
"""
import torch
from .reconstruction_only import is_pose_parameter


def restoration_parameter(name):
    return not (is_pose_parameter(name) or name.startswith(('core.feature_mid.','core.feature_last.')))


def backward_primary_geometry(geometry,secondary,patches,named_parameters,ratio=.5):
    if not 0<ratio<=1:raise ValueError('Expected a secondary-to-geometry gradient ratio in (0,1]')
    patches=tuple(patches)
    geo=torch.autograd.grad(geometry,patches,retain_graph=True,allow_unused=True)
    other=torch.autograd.grad(secondary,patches,retain_graph=True,allow_unused=True)
    norm=lambda values:sum((v.detach().float().square().sum() for v in values if v is not None),geometry.detach().new_zeros(())).sqrt()
    gn,on=norm(geo),norm(other)
    scale=torch.where(gn>1e-12,(ratio*gn/on.clamp_min(1e-12)).clamp_max(1.),gn.new_ones(())).detach()
    handles=[]
    try:
        for name,param in named_parameters:
            if param.requires_grad and restoration_parameter(name):handles.append(param.register_hook(lambda gradient,s=scale:gradient*s))
        secondary.backward(retain_graph=True)
    finally:
        for handle in handles:handle.remove()
    geometry.backward()
    return dict(geometry_patch_grad_norm=gn,secondary_patch_grad_norm=on,
                restoration_secondary_scale=scale,effective_secondary_patch_ratio=scale*on/gn.clamp_min(1e-12))
