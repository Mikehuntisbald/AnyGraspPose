"""Frozen paired diagnostics: spatial identity, coupled geometry, history content."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from lip.jepa.losses import normalize, feature_error
from .recovery_focus import camera_disagreement


@torch.no_grad()
def feature_diagnostics(output, teacher, weight, proxy=False, middle=False):
    if not weight.any(): return None
    target = (teacher.proxy_mid if proxy else teacher.real_mid) if middle else (teacher.proxy_last if proxy else teacher.real_last)
    prediction = output["f_mid_predicted" if middle else "f_predicted"]
    candidates = (teacher.support_label.nan_to_num() >= .9) if proxy else ((teacher.visible_weight+teacher.hidden_real_weight)>0)
    if (candidates.sum(1)<2).any(): return None
    p, t = normalize(prediction), normalize(target)
    w = weight.float(); c = candidates.float()
    mean = lambda x: float((x*w).sum()/w.sum())
    pm = (p*c[..., None]).sum(1, keepdim=True)/c.sum(1)[:, None, None].clamp_min(1)
    tm = (t*c[..., None]).sum(1, keepdim=True)/c.sum(1)[:, None, None].clamp_min(1)
    pc, tc = p-pm, t-tm
    pvar = ((pc.square().mean(-1))*c).sum()/c.sum().clamp_min(1)
    tvar = ((tc.square().mean(-1))*c).sum()/c.sum().clamp_min(1)
    sim = F.normalize(p, dim=-1)@F.normalize(t, dim=-1).transpose(-1, -2)
    sim = sim.masked_fill(~candidates[:, None], -1e4)
    found = sim.argmax(-1); index = torch.arange(p.shape[1], device=p.device)[None]
    displacement = ((found%16-index%16).square()+(found//16-index//16).square()).float().sqrt()*14
    wrong = sim.masked_fill(torch.eye(p.shape[1], device=p.device, dtype=torch.bool)[None], -1e4).amax(-1)
    shuffled = p.clone()
    for lane in range(len(p)):
        ids = candidates[lane].nonzero().flatten()
        if len(ids)>1: shuffled[lane, ids] = p[lane, ids.roll(1)]
    oracle=tm.expand_as(t)
    oracle_similarity=(F.normalize(tm,dim=-1)@F.normalize(t,dim=-1).transpose(-1,-2)).masked_fill(~candidates[:,None],-1e4)
    oracle_found=oracle_similarity.argmax(-1).expand(-1,p.shape[1])
    return dict(patches=float(w.sum()), candidates=float(c.sum()/len(c)),
        teacher_mean_cosine=mean(F.cosine_similarity(oracle,t,dim=-1)),
        teacher_mean_retrieval=mean((oracle_found==index).float()),
        feature_loss=mean(feature_error(prediction,target)),
        teacher_mean_feature_loss=mean(feature_error(oracle,target)),
        cosine=mean(F.cosine_similarity(p, t, dim=-1)),
        shuffled_cosine=mean(F.cosine_similarity(shuffled, t, dim=-1)),
        constant_cosine=mean(F.cosine_similarity(pm.expand_as(p), t, dim=-1)),
        centered_cosine=mean(F.cosine_similarity(pc, tc, dim=-1)),
        spatial_variance_ratio=float(pvar/tvar.clamp_min(1e-9)),
        retrieval_top1=mean((found==index).float()), retrieval_error_px=mean(displacement),
        correct_minus_best_wrong=mean(sim.diagonal(dim1=-2,dim2=-1)-wrong))


@torch.no_grad()
def geometry_diagnostics(output, teacher, weight, diameter):
    if not weight.any(): return None
    w = weight.float(); mean = lambda x: float((x*w).sum()/w.sum())
    xyz = output['surface_xyz'].float(); target = teacher.surface_xyz.float()
    camera = torch.einsum('bij,bjhw->bihw', teacher.camera_rotation, xyz)+teacher.camera_translation_d[:, :, None, None]
    depth = output['surface_depth_residual'].float()+teacher.base_depth_d[:, None, None, None]
    consistency = camera_disagreement(output, teacher)
    rays = teacher.camera_rays
    # Rays were built at integer pixel centers; inverse slope is crop focal length.
    fx = 1/(rays[:, 0, 0, 1]-rays[:, 0, 0, 0])
    fy = 1/(rays[:, 1, 1, 0]-rays[:, 1, 0, 0])
    projected = camera[:, :2]/camera[:, 2:3].clamp_min(1e-4)-rays[:, :2]
    projected = projected*torch.stack((fx, fy), 1)[:, :, None, None]
    probability = output['geometry_valid_logits'].float().sigmoid()
    return dict(pixels=float(w.sum()),
        xyz_mm=mean((xyz-target).norm(dim=1,keepdim=True)*diameter*1000),
        zero_xyz_mm=mean(target.norm(dim=1,keepdim=True)*diameter*1000),
        depth_mm=mean((output['surface_depth_m']-teacher.surface_depth_m).abs()*1000),
        camera_consistency_mm=mean(consistency.norm(dim=1,keepdim=True)*diameter*1000),
        xyz_depth_inconsistency_mm=mean((camera[:, 2:3]-depth).abs()*diameter*1000),
        xyz_reprojection_px=mean(projected.norm(dim=1,keepdim=True)),
        behind_camera_fraction=mean((camera[:, 2:3]<=0).float()),
        valid_probability=mean(probability), valid_brier=mean((probability-1).square()))


def scramble_history(memory):
    """Rotate content among valid slots; preserve positions, ages and validity.

    This is a spatial association intervention, not an unrelated-object history
    test. No labels are used. The ordinary writer/memory remain unmodified.
    """
    if memory is None: return None
    def corrupt(record):
        features = record.features.clone()
        for lane in range(len(features)):
            ids = record.valid[lane].nonzero().flatten()
            if len(ids)>1: features[lane, ids] = features[lane, ids.roll(max(1,len(ids)//2))]
        return replace(record, features=features)
    return replace(memory, objects=tuple(corrupt(r) for r in memory.objects),
                   contexts=tuple(corrupt(r) for r in memory.contexts))
