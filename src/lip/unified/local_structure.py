"""Teacher-only local relations. No network/input/pose-path changes.

Both endpoints use the same target source. Unsupervised predictions are never
read: detached teacher features supply anchors there, only inside the loss.
"""
import math
import torch
from torch.nn import functional as F
from lip.jepa.losses import normalize, sample_mean

LOCAL_METRICS = tuple(f'{term}_{source}_{layer}' for source in ('real', 'proxy')
                      for layer in ('mid', 'last') for term in ('local_difference', 'local_correspondence'))


def _grids(p, t, query, candidates):
    side = math.isqrt(p.shape[1])
    if side * side != p.shape[1]:
        raise ValueError('Local relations require a square patch grid')
    valid = candidates.bool().reshape(-1, side, side)
    q = query.float().reshape(-1, side, side) * valid
    target = t.detach().reshape(-1, side, side, t.shape[-1])
    prediction = p.reshape_as(target)
    completed = torch.where((q > 0)[..., None], prediction, target)
    return completed, target, q, valid


def difference_terms(p, t, query, candidates, scales=(1, 2, 4)):
    """Inputs are already normalized per patch, before spatial subtraction.

    Equal mass per valid sample and per active scale. The RMS scale is from
    detached target differences, with a floor; low-contrast noise is not inflated.
    All intermediate grid sites must be valid, preventing long edges across holes.
    """
    p, t, q, valid = _grids(p, t, query, candidates)
    side = p.shape[1]; losses = []; statistics = []; active_scales = []
    for distance in scales:
        predicted = []; targets = []; weights = []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if distance >= side:
                continue
            y0, y1 = 0, side - dy * distance
            x0, x1 = max(0, -dx * distance), side - max(0, dx * distance)
            a = (slice(None), slice(y0, y1), slice(x0, x1))
            b = (slice(None), slice(y0 + dy * distance, y1 + dy * distance),
                 slice(x0 + dx * distance, x1 + dx * distance))
            known = valid[a] & valid[b]
            for k in range(1, distance):
                known = known & valid[:, y0 + dy*k:y1 + dy*k, x0 + dx*k:x1 + dx*k]
            weights.append((torch.maximum(q[a], q[b]) * known).flatten(1))
            predicted.append((p[a] - p[b]).flatten(1, 2))
            targets.append((t[a] - t[b]).flatten(1, 2))
        if not weights:
            continue
        w = torch.cat(weights, 1); pd = torch.cat(predicted, 1); td = torch.cat(targets, 1)
        mass = w.sum(1); active = mass > 0
        energy = (td.square().mean(-1) * w).sum(1) / mass.clamp_min(1)
        scale = energy.sqrt().clamp_min(.1)[:, None, None]
        error = F.smooth_l1_loss(pd / scale, td / scale, beta=.1, reduction='none').mean(-1)
        per_sample = (error * w).sum(1) / mass.clamp_min(1)
        losses.append(per_sample); active_scales.append(active)
        squared = ((pd-td).square().mean(-1) * w).sum(1) / mass.clamp_min(1)
        pred_energy = (pd.square().mean(-1) * w).sum(1) / mass.clamp_min(1)
        statistics.append((distance, sample_mean(error, w)[0],
            ((squared / energy.clamp_min(.01)).sqrt() * active).sum() / active.sum().clamp_min(1),
            ((pred_energy / energy.clamp_min(.01)).sqrt() * active).sum() / active.sum().clamp_min(1),
            mass.sum()))
    if not losses:
        return p.sum()*0, statistics
    values = torch.stack(losses, 1); active = torch.stack(active_scales, 1)
    return sample_mean(values, active)[0], statistics


def local_correspondence(p, t, query, candidates, radius=2, temperature=.1):
    """Match local residual descriptors within a 5x5 window, with soft targets.

    Similar teacher sites remain alternatives; no synthetic one-hot negatives.
    Means contain only valid same-source sites. Visible predictions have no loss
    gradient. Target anchors cannot enter the student forward.
    """
    p, t, q, valid = _grids(p, t, query, candidates)
    side = p.shape[1]; kernel = 2 * radius + 1
    w = valid[:, None].float()
    denominator = F.avg_pool2d(w, kernel, 1, radius).clamp_min(1e-6)
    def residual(x):
        grid = x.permute(0, 3, 1, 2)
        mean = F.avg_pool2d(grid*w, kernel, 1, radius) / denominator
        return (grid-mean).flatten(2).transpose(1, 2)
    pr = F.normalize(residual(p), dim=-1, eps=.1)
    tr = F.normalize(residual(t), dim=-1, eps=.1)
    ids = torch.arange(side*side, device=p.device)
    window = ((ids[:, None]//side-ids[None, :]//side).abs() <= radius) & \
             ((ids[:, None]%side-ids[None, :]%side).abs() <= radius)
    keys = window[None] & valid.flatten(1)[:, None]
    logits = (pr @ tr.transpose(-1, -2) / temperature).masked_fill(~keys, -1e4)
    teacher_logits = (tr @ tr.transpose(-1, -2) / temperature).masked_fill(~keys, -1e4)
    truth = teacher_logits.softmax(-1)
    kl = (truth*(teacher_logits.log_softmax(-1)-logits.log_softmax(-1))).sum(-1)
    eligible = q.flatten(1) * (keys.sum(-1) >= 2)
    return sample_mean(kl, eligible)[0]


def local_structure_loss(output, target, weights):
    total = output['f_predicted'].float().sum()*0; parts = {}
    sources = [('real', target.real_mid, target.real_last, target.hidden_real_weight,
                target.visible_weight+target.hidden_real_weight, 1.),
               ('proxy', target.proxy_mid, target.proxy_last, target.proxy_weight,
                target.support_label.nan_to_num() >= .9, weights['cad_feature'])]
    with torch.autocast(output['f_predicted'].device.type, enabled=False):
        for source, mid, last, query, candidates, factor in sources:
            for layer, pred, teacher, coefficient in (
                ('mid', output['f_mid_predicted'], mid, weights.get('feature_mid_weight', .25)),
                ('last', output['f_predicted'], last, weights.get('feature_last_weight', 1.))):
                p, t = normalize(pred), normalize(teacher.detach())
                diff, _ = difference_terms(p, t, query, candidates)
                corr = local_correspondence(p, t, query, candidates)
                total = total + factor*coefficient*(weights.get('local_difference', 0.)*diff +
                                                    weights.get('local_correspondence', 0.)*corr)
                parts[f'local_difference_{source}_{layer}'] = diff.detach()
                parts[f'local_correspondence_{source}_{layer}'] = corr.detach()
    return total, parts


@torch.no_grad()
def local_diagnostics(prediction, target, query, candidates):
    with torch.autocast(prediction.device.type, enabled=False):
        p, t = normalize(prediction), normalize(target.detach())
        _, stats = difference_terms(p, t, query, candidates)
        result = {'local_correspondence_kl': float(local_correspondence(p, t, query, candidates))}
        for distance, loss, relative, amplitude, pairs in stats:
            result.update({f'diff{distance}_loss': float(loss), f'diff{distance}_relative_rmse': float(relative),
                           f'diff{distance}_amplitude_ratio': float(amplitude), f'diff{distance}_pairs': float(pairs)})
    return result
