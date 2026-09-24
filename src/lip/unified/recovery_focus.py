"""Spatially discriminative completion and coupled camera geometry.

All targets and camera transforms are teacher-only. Predictions at unmasked
visible locations receive no completion gradients. Similar teacher patches are
soft alternatives, never forced hard negatives (e.g. textureless/symmetric CAD).
"""
import torch
from torch.nn import functional as F
from lip.jepa.losses import normalize, sample_mean

FOCUS_METRICS = ('spatial_centered_real', 'spatial_centered_proxy',
                 'correspondence_real', 'correspondence_proxy',
                 'camera_consistency_real', 'camera_consistency_proxy')


def masked_mean(value, weight):
    return sample_mean(value.flatten(1), weight.expand_as(value).float().flatten(1))[0]


def spatial_terms(prediction, target, query_weight, candidate_weight, temperature=.1):
    """Center over supervised queries; keys are detached same-source targets.

    Centering over queries (not all predictions) avoids gradients into visible
    predictions. The teacher bank may include visible locations as alternatives.
    Its pairwise distribution preserves teacher ambiguity instead of inventing
    point identity from position alone. Fewer than two queries: no spatial loss.
    """
    p, t = normalize(prediction), normalize(target.detach())
    q = query_weight.float(); keys = candidate_weight.bool()
    count = q.sum(1, keepdim=True)
    q = q * (count >= 2)
    den = q.sum(1, keepdim=True).clamp_min(1)
    pm = (p*q[..., None]).sum(1, keepdim=True)/den[..., None]
    tm = (t*q[..., None]).sum(1, keepdim=True)/den[..., None]
    pc, tc = p-pm, t-tm
    scale = ((tc.square().mean(-1)*q).sum(1, keepdim=True)/den).sqrt().clamp_min(.1)
    centered = F.smooth_l1_loss(pc/scale[..., None], tc/scale[..., None], beta=.1,
                              reduction='none').mean(-1)
    key_mean = (t*keys[..., None]).sum(1, keepdim=True)/keys.sum(1)[:, None, None].clamp_min(1)
    k = F.normalize(t-key_mean, dim=-1)
    # Restore teacher query-to-bank offset after removing prediction's object mean.
    query = F.normalize(pc + tm-key_mean, dim=-1)
    teacher_query = F.normalize(t-key_mean, dim=-1)
    logits = (query @ k.transpose(-1, -2))/temperature
    teacher_logits = (teacher_query @ k.transpose(-1, -2))/temperature
    logits = logits.masked_fill(~keys[:, None], -1e4)
    teacher_logits = teacher_logits.masked_fill(~keys[:, None], -1e4)
    truth = teacher_logits.softmax(-1)
    kl = (truth * (teacher_logits.log_softmax(-1)-logits.log_softmax(-1))).sum(-1)
    eligible = q * (keys.sum(1, keepdim=True) >= 2)
    return masked_mean(centered, q), masked_mean(kl, eligible)


def camera_disagreement(output, target):
    """Full XYZ coupling, including camera X/Y, in object-diameter units."""
    if target.camera_rotation is None or target.camera_rays is None:
        raise ValueError('Recovery focus requires isolated camera teacher metadata')
    with torch.autocast(output['surface_xyz'].device.type, enabled=False):
        xyz = output['surface_xyz'].float()
        camera = torch.einsum('bij,bjhw->bihw', target.camera_rotation.float(), xyz)
        camera = camera + target.camera_translation_d[:, :, None, None]
        depth_d = output['surface_depth_residual'].float() + target.base_depth_d[:, None, None, None]
        return camera - target.camera_rays.float()*depth_d


def focus_loss(output, target, weights, real_only=False):
    zero = output['f_predicted'].float().sum()*0
    parts = {key: zero for key in FOCUS_METRICS}
    total = zero
    feature_sources = [('real', target.real_last, target.real_mid, target.hidden_real_weight,
                        target.visible_weight+target.hidden_real_weight, 1.)]
    if not real_only:
        feature_sources.append(('proxy', target.proxy_last, target.proxy_mid, target.proxy_weight,
                                target.support_label.nan_to_num() >= .9, weights['cad_feature']))
    wm,wl=weights.get("feature_mid_weight",.25),weights.get("feature_last_weight",1.)
    for name, last, mid, query, candidates, factor in feature_sources:
        a, b = spatial_terms(output['f_predicted'], last, query, candidates)
        am, bm = spatial_terms(output['f_mid_predicted'], mid, query, candidates)
        parts['spatial_centered_'+name] = wl*a+wm*am
        parts['correspondence_'+name] = wl*b+wm*bm
        total = total + factor*(weights.get('spatial_centered', 0.)*(wl*a+wm*am)
                                + weights.get('spatial_correspondence', 0.)*(wl*b+wm*bm))
    residual = camera_disagreement(output, target)
    consistency = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.05,
                                   reduction='none').mean(1, keepdim=True)
    for name, mask in [('real', target.geometry_real_weight), ('proxy', target.geometry_proxy_weight)]:
        if real_only and name == 'proxy': continue
        value = masked_mean(consistency, mask)
        parts['camera_consistency_'+name] = value
        total = total + weights.get('camera_consistency', 0.) * (1. if name == 'real' else weights['cad_feature']) * value
    return total, {k: v.detach() for k, v in parts.items()}
