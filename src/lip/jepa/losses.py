"""All loss weights come from label validity, never predicted confidence."""
import torch
from torch.nn import functional as F


def normalize(x):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6)


def sample_mean(values, weights, distributed=False):
    weights = weights.float()
    den = weights.sum(-1)
    valid = den > 0
    means = (values * weights).sum(-1) / den.clamp_min(1e-12)
    count = valid.sum()
    scale = 1
    if distributed:
        import torch.distributed as dist
        count = count.clone()
        dist.all_reduce(count)
        scale = dist.get_world_size()
    return (means * valid).sum() * scale / count.clamp_min(1), count


def feature_error(prediction, target):
    p, t = normalize(prediction), normalize(target.detach())
    return F.smooth_l1_loss(p, t, reduction='none').mean(-1) + .25 * (1 - F.cosine_similarity(p, t, dim=-1))


def completion_loss(output, targets, distributed=False):
    valid = targets.feature_target_valid.bool()
    hidden = targets.added_occlusion_fraction.float()
    weights = valid * (hidden + .1 * (1 - hidden))
    last, count = sample_mean(feature_error(output.f_predicted, targets.teacher_features_last), weights, distributed)
    mid, _ = sample_mean(feature_error(output.f_mid_predicted, targets.teacher_features_mid), weights, distributed)
    mse = (normalize(output.f_predicted).detach() - normalize(targets.teacher_features_last.detach())).square().mean(-1)
    unc, _ = sample_mean(F.smooth_l1_loss(output.log_feature_error.float(), (mse + 1e-6).log().clamp(-14, 5), reduction='none'), weights, distributed)
    zero = output.f_predicted.float().sum() * 0
    mask = zero
    for logits, label in ((output.evidence_logits, targets.object_visible_target),
                           (output.support_logits, targets.object_support_target)):
        if label is not None:
            known = torch.isfinite(label)
            term, _ = sample_mean(F.binary_cross_entropy_with_logits(logits.float(), label.nan_to_num().float(), reduction='none'), known, distributed)
            mask = mask + term
    loss = last + .25 * mid + .1 * mask + .05 * unc
    return dict(loss=loss, latent=last, mid=mid, mask=mask, uncertainty=unc, valid_samples=count,
                hidden_targets=(valid & (hidden > .5)).sum())
