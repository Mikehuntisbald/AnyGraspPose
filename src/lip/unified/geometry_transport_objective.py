"""Geometry-only training, disjoint real/proxy targets, no pose/DINO loss."""
import torch
from torch.nn import functional as F
from .cad_transport import transport_targets
from .recovery_focus import masked_mean, camera_disagreement
from .surface_normals import normal_terms


def objective(output, target, observation, visible_mask, crop_k, use_transport):
    beta = .02
    def distance(a, b):
        return F.smooth_l1_loss(a.float(), b.float(), beta=beta, reduction='none').mean(1, keepdim=True)
    # Preserve REAL measurements as their own target, never relabel them as CAD.
    measured = observation.measured_depth_m.float()
    camera = target.camera_rays*measured/observation.diameter[:, None, None, None]-target.camera_translation_d[:, :, None, None]
    xyz_visible = torch.einsum('bji,bjhw->bihw', target.camera_rotation, camera).detach()
    visible = visible_mask & (measured > 0) & torch.isfinite(measured) & (xyz_visible.square().sum(1, keepdim=True) < 1.)
    xyz_target = torch.where(visible, xyz_visible, target.surface_xyz)
    masks = [('real', target.geometry_real_weight, 1.), ('proxy', target.geometry_proxy_weight, .5)]
    def surface_loss(xyz, depth):
        loss = xyz.sum()*0
        disagreement = camera_disagreement(dict(surface_xyz=xyz, surface_depth_residual=depth), target)
        for _, mask, factor in masks:
            loss = loss+factor*(masked_mean(distance(xyz, target.surface_xyz), mask)
                                + masked_mean(distance(depth, target.surface_depth_residual), mask)
                                + .5*masked_mean(distance(disagreement, torch.zeros_like(disagreement)), mask))
        return loss+.5*masked_mean(distance(xyz, xyz_visible), visible)
    loss = surface_loss(output['surface_xyz'], output['surface_depth_residual'])
    if 'coarse_surface' in output:
        coarse = output['coarse_surface']
        loss = loss+.2*surface_loss(coarse[:, :3], coarse[:, 3:4])
    known = torch.isfinite(target.geometry_valid_label)
    validity = masked_mean(F.binary_cross_entropy_with_logits(output['geometry_valid_logits'].float(), target.geometry_valid_label.nan_to_num(), reduction='none'), known)
    loss = loss+.05*validity
    for key, label in [('evidence_logits', target.visible_label), ('support_logits', target.support_label)]:
        loss = loss+.05*masked_mean(F.binary_cross_entropy_with_logits(output[key].float(), label.nan_to_num(), reduction='none'), torch.isfinite(label))
    for _, mask, factor in masks:
        error, valid, _ = normal_terms(output['surface_xyz'], target.surface_xyz, mask, 2)
        loss = loss+.02*factor*masked_mean(error, valid)
    metrics = {}
    scale = observation.diameter[:, None, None, None]*1000
    for name, mask, _ in masks:
        metrics[name+'_pixels'] = mask.sum().detach()
        metrics[name+'_xyz_mm'] = masked_mean((output['surface_xyz']-target.surface_xyz).norm(dim=1, keepdim=True)*scale, mask).detach()
        metrics[name+'_depth_mm'] = masked_mean((output['surface_depth_residual']-target.surface_depth_residual).abs()*scale, mask).detach()
    metrics['visible_xyz_mm'] = masked_mean((output['surface_xyz']-xyz_visible).norm(dim=1, keepdim=True)*scale, visible).detach()
    if use_transport:
        truth = transport_targets(xyz_target, observation.geometry_image, observation.base, observation.diameter, crop_k, observation.cad_valid)
        domain = target.geometry_weight | visible
        eligible = truth['supported'] & domain
        flow = masked_mean(distance(output['transport_flow']/224., truth['flow']/224.), eligible)
        gate = masked_mean(F.binary_cross_entropy_with_logits(output['transport_gate_logits'], truth['supported'].float(), reduction='none'), domain)
        recovered = output['transport_surface']
        transported_xyz = masked_mean(distance(recovered[:, :3], xyz_target), eligible)
        transported_depth = masked_mean(distance(recovered[:, 3:4], target.surface_depth_residual), eligible & target.geometry_weight)
        loss = loss+.25*flow+.1*gate+.5*(transported_xyz+transported_depth)
        metrics.update(flow_pixels=eligible.sum().detach(), transport_supported_fraction=(eligible.sum()/domain.sum().clamp_min(1)).detach(),
                       flow_epe=masked_mean((output['transport_flow']-truth['flow']).norm(dim=1, keepdim=True), eligible).detach(),
                       transport_gate=masked_mean(output['transport_gate'], domain).detach())
    return loss, metrics
