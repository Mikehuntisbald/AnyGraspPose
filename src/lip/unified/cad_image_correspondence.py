"""CAD -> image matches from the SAME JEPA/DPT atlas similarity.

Inspired by GoTrack's explicit image endpoints and visibility, not a reproduction
of its template decoder. No base-projection prior or teacher in this readout.
The existing image -> CAD reconstruction continues to use the same q/k tensors.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def image_grid(height, width, stride, device):
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    return torch.stack((x, y), -1).float().reshape(-1, 2)*stride+(stride-1)/2


def sample_points(image, uv, mode='bilinear'):
    h, w = image.shape[-2:]
    grid = 2*(uv.float()+.5)/uv.new_tensor([w, h])-1
    return F.grid_sample(image.float(), grid[:, None], mode=mode, align_corners=False)[:, :, 0].transpose(1, 2)


def anchored_flow_loss(output,labels):
    difference=(output['cad_flow_uv'].float()-labels['uv'].float())/224.
    error=F.smooth_l1_loss(difference,torch.zeros_like(difference),beta=1/224.,reduction='none').sum(-1)
    loss=error.sum()*0;metrics={}
    for name,factor in [('observed',1.),('real',1.),('proxy',.5)]:
        mask=labels[name];count=mask.sum(-1);eligible=count>0
        value=(error*mask).sum(-1)/count.clamp_min(1)
        loss+=factor*(value*eligible).sum()/eligible.sum().clamp_min(1)
        epe=(output['cad_flow_uv'].detach()-labels['uv']).norm(dim=-1)
        metrics['flow_'+name+'_epe']=((epe*mask).sum(-1)/count.clamp_min(1)*eligible).sum()/eligible.sum().clamp_min(1)
    return loss,metrics


class CADImageReadout(nn.Module):
    def __init__(self, width=32, points=512, stride=2):
        super().__init__()
        self.points, self.stride = points, stride
        self.visibility = nn.Sequential(nn.Linear(2*width+3, 64), nn.GELU(), nn.Linear(64, 2))

    @torch.autocast('cuda', enabled=False)
    def forward(self, atlas, image_shape, estimated_uv=None, prior_sigma=None):
        b, _, width = atlas['atlas_query'].shape
        h, w = image_shape
        qmap = atlas['atlas_query'].transpose(1, 2).reshape(b, width, h, w).float()
        qmap = F.normalize(F.avg_pool2d(qmap, self.stride), dim=1)
        q = qmap.flatten(2).transpose(1, 2)
        ids = torch.linspace(0, atlas['atlas_keys'].shape[1]-1, min(self.points, atlas['atlas_keys'].shape[1]), device=q.device).long()
        keys = atlas['atlas_keys'][:, ids].float()
        available = atlas['atlas_available'][:, ids]
        scores = (keys @ q.transpose(-1, -2))/atlas['atlas_temperature']
        grid = image_grid(*qmap.shape[-2:], self.stride, q.device)
        if prior_sigma is not None:
            if estimated_uv is None or prior_sigma<=0:raise ValueError('Explicit estimated projection and positive sigma required')
            scores=scores-(estimated_uv[:,ids,None].float()-grid[None,None]).square().sum(-1)/(2*prior_sigma**2)
        # Local soft argmax preserves subpixel endpoints without averaging distant
        # ambiguous modes. The global spatial CE supplies gradients to far errors.
        peak = scores.detach().argmax(-1)
        peak_uv = grid[peak]
        near = (grid[None, None]-peak_uv[:, :, None]).abs().amax(-1) <= self.stride
        probability = scores.masked_fill(~near, -1e4).softmax(-1)
        uv = probability @ grid
        matched = probability @ q
        logp = scores.log_softmax(-1)
        entropy = -(logp.exp()*logp).sum(-1)/math.log(len(grid))
        top = scores.topk(2, dim=-1).values
        summary = torch.stack((entropy, top[..., 0]-top[..., 1], logp.max(-1).values.exp()), -1)
        logits = self.visibility(torch.cat((keys, matched, summary), -1))
        confidence = logits.sigmoid().prod(-1)*(1-entropy).clamp_min(0)*available
        result=dict(cad_image_ids=ids, cad_image_xyz=atlas['atlas_xyz'][:, ids], cad_image_uv=uv,
                    cad_image_available=available, cad_image_scores=scores, cad_image_grid=grid,
                    cad_image_support_logits=logits[..., 0], cad_image_visible_logits=logits[..., 1],
                    cad_image_confidence=confidence, cad_image_entropy=entropy,
                    cad_image_stride=self.stride)
        if hasattr(self,'anchor_flow'):
            geometry=atlas['atlas_geometry'][:,ids].float()
            reference=geometry[:,:,15:17]*w
            # Frozen-backbone readout test. Only this small MLP learns; sampled
            # features include the restored XYZ/depth and the shared DPT query.
            source_query=sample_points(atlas['atlas_query'].transpose(1,2).reshape(b,width,h,w),reference)
            recovered=sample_points(atlas['atlas_fallback'],reference)
            feature=torch.cat((source_query,keys,q.mean(1)[:,None].expand_as(keys),geometry,recovered),-1).detach()
            displacement=56*self.anchor_flow(feature).tanh()
            result.update(cad_flow_uv=reference+displacement,cad_flow_delta=displacement,cad_flow_reference=reference)
        return result


@torch.no_grad()
@torch.autocast('cuda', enabled=False)
def image_correspondence_targets(output, target, crop_k, diameter, visible):
    """GT projections are labels ONLY. Reject back surfaces and ambiguous edges.

Coordinates are canonical / diameter. Intrinsics already include the crop.
Real depth labels are never changed; point identity labels always refer to CAD.
"""
    xyz = output['cad_image_xyz'].detach().float()
    camera_d = xyz @ target.camera_rotation.float().transpose(-1, -2)+target.camera_translation_d[:, None].float()
    projected = camera_d @ crop_k.float().transpose(-1, -2)
    uv = projected[..., :2]/projected[..., 2:].clamp_min(1e-6)
    h, w = target.cad_geometry_valid.shape[-2:]
    inside = (uv >= 2).all(-1) & (uv < uv.new_tensor([w-2, h-2])).all(-1) & (camera_d[..., 2] > 0)
    valid = target.cad_geometry_valid
    interior = -F.max_pool2d(-valid.float(), 5, 1, 2) > .999
    zbuffer = sample_points(target.cad_geometry_depth_m, uv)[..., 0]
    z = camera_d[..., 2]*diameter[:, None]
    front = (z-zbuffer).abs() <= .003
    xyz_at_pixel = sample_points(target.cad_geometry_xyz, uv)
    same_surface = (xyz-xyz_at_pixel).norm(dim=-1) <= .02
    silhouette = sample_points(valid.float(), uv, 'nearest')[..., 0] > .5
    interior = sample_points(interior.float(), uv, 'nearest')[..., 0] > .5
    support = inside & interior & front & same_surface
    # Border/intersection uncertainty is not a negative label. Clear back surfaces
    # are valid support negatives, including those projecting onto the silhouette.
    clear_negative = ~inside | ~silhouette | (z > zbuffer+.01)
    known_support = (support | clear_negative) & output['cad_image_available']
    eligible = getattr(target, 'real_geometry_eligible', None)
    vis = visible if eligible is None else visible & eligible
    # Erode visible/hidden boundaries so a partially occluded sample is unknown.
    erode = lambda x: -F.max_pool2d(-x.float(), 3, 1, 1) > .999
    current_visible = sample_points(erode(vis).float(), uv, 'nearest')[..., 0] > .5
    current_hidden = sample_points(erode(~visible).float(), uv, 'nearest')[..., 0] > .5
    real = sample_points(target.geometry_real_weight.float(), uv, 'nearest')[..., 0] > .5
    proxy = sample_points(target.geometry_proxy_weight.float(), uv, 'nearest')[..., 0] > .5
    available = output['cad_image_available']
    return dict(uv=uv, support=support, known_support=known_support,
                visible=support & current_visible, known_visible=support & (current_visible | current_hidden) & available,
                real=support & real & available, proxy=support & proxy & available,
                observed=support & current_visible & available)


@torch.autocast('cuda', enabled=False)
def image_correspondence_loss(output, labels, balanced_visibility=False, visibility_weight=.1):
    logits = output['cad_image_scores'].float()
    # Gaussian endpoint labels with exact pixel-center convention. No estimated
    # pose prior in this CE: the matching features must localize the CAD point.
    d2 = (labels['uv'][:, :, None]-output['cad_image_grid'][None, None]).square().sum(-1)
    heat = (-d2/(2*1.5**2)).softmax(-1).detach()
    ce = -(heat*logits.log_softmax(-1)).sum(-1)
    def mean_by_example(value, mask):
        counts = mask.sum(-1)
        per = (value*mask).sum(-1)/counts.clamp_min(1)
        return (per*(counts > 0)).sum()/(counts > 0).sum().clamp_min(1)
    loss = logits.sum()*0
    metrics = {}
    epe = (output['cad_image_uv'].detach()-labels['uv']).norm(dim=-1)
    for name, factor in [('observed', 1.), ('real', 1.), ('proxy', .5)]:
        mask = labels[name]
        loss = loss+factor*mean_by_example(ce, mask)
        metrics['image_'+name+'_epe'] = mean_by_example(epe, mask).detach()
        metrics['image_'+name+'_count'] = mask.sum().detach()
    for name in ('support', 'visible'):
        error = F.binary_cross_entropy_with_logits(output['cad_image_'+name+'_logits'].float(), labels[name].float(), reduction='none')
        if balanced_visibility:
            binary=balanced_binary_loss(output['cad_image_'+name+'_logits'].float(),labels[name],labels['known_'+name])
        else:binary=mean_by_example(error, labels['known_'+name])
        loss = loss+visibility_weight*binary
    return loss, metrics


def balanced_binary_loss(logits, label, known):
    """Equal positive/negative class mass within each nonempty example."""
    error=F.binary_cross_entropy_with_logits(logits,label.float(),reduction='none')
    masks=torch.stack((known&label,known&~label),1)
    count=masks.sum(-1)
    mean=(error[:,None]*masks).sum(-1)/count.clamp_min(1)
    active=count>0
    per=(mean*active).sum(-1)/active.sum(-1).clamp_min(1)
    return (per*active.any(-1)).sum()/active.any(-1).sum().clamp_min(1)


def solve_correspondences(xyz, uv, confidence, crop_k, base, seed=42):
    """Prediction-only PnP; deterministic RANSAC and guarded fallback, no GT args."""
    import cv2
    import numpy as np
    xyz, uv, confidence, crop_k, base = [np.asarray(x, dtype=np.float64) for x in (xyz, uv, confidence, crop_k, base)]
    ok = np.isfinite(xyz).all(-1) & np.isfinite(uv).all(-1) & np.isfinite(confidence) & (confidence > 1e-7)
    ids = np.flatnonzero(ok)
    if len(ids) > 1024: ids = ids[np.argsort(-confidence[ids], kind='stable')[:1024]]
    receipt = dict(accepted=False, correspondences=len(ids), inliers=0, inlier_ratio=0.)
    if len(ids) < 12 or np.linalg.svd(xyz[ids]-xyz[ids].mean(0), compute_uv=False)[1] < 1e-5:
        return base.copy(), receipt

    cv2.setRNGSeed(seed)
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(np.ascontiguousarray(xyz[ids]), np.ascontiguousarray(uv[ids]), crop_k, None,
            iterationsCount=1000, reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None: return base.copy(), receipt
        receipt.update(inliers=len(inliers), inlier_ratio=len(inliers)/len(ids))
        if len(inliers) < 12 or receipt['inlier_ratio'] < .15: return base.copy(), receipt
        chosen = ids[inliers[:, 0]]
        if np.linalg.svd(uv[chosen]-uv[chosen].mean(0), compute_uv=False)[1]/math.sqrt(len(chosen)) < 4:
            return base.copy(), receipt
        rvec, tvec = cv2.solvePnPRefineLM(np.ascontiguousarray(xyz[chosen]), np.ascontiguousarray(uv[chosen]), crop_k, None, rvec, tvec)
        rotation = cv2.Rodrigues(rvec)[0]
        predicted = xyz[chosen]@rotation.T+tvec.reshape(3)
        projected = predicted@crop_k.T
        epe = np.linalg.norm(projected[:, :2]/projected[:, 2:]-uv[chosen], axis=-1)
        if not np.isfinite(epe).all() or (predicted[:, 2] <= 0).any() or np.median(epe) > 3:
            return base.copy(), receipt
        pose = base.copy(); pose[:3, :3] = rotation; pose[:3, 3] = tvec.reshape(3)
        receipt.update(accepted=True, inlier_median_px=float(np.median(epe)))
        return pose, receipt
    except cv2.error:
        receipt['opencv_failure'] = True
        return base.copy(), receipt

def solve_rgbd_correspondences(xyz, camera, confidence, base):
    """Robust3D-3D diagnostic; measured/recovered reliability is supplied by caller."""
    import numpy as np
    xyz,camera,confidence,base=[np.asarray(x,dtype=np.float64) for x in (xyz,camera,confidence,base)]
    good=np.isfinite(xyz).all(-1)&np.isfinite(camera).all(-1)&np.isfinite(confidence)&(confidence>1e-7)&(camera[:,2]>0)
    x,y,w=xyz[good],camera[good],confidence[good]
    receipt=dict(accepted=False,correspondences=len(x),inliers=0,inlier_ratio=0.)
    if len(x)<12 or np.linalg.svd(x-x.mean(0),compute_uv=False)[1]<1e-5:return base.copy(),receipt
    pose=base.copy()
    for _ in range(6):
        residual=np.linalg.norm(x@pose[:3,:3].T+pose[:3,3]-y,axis=-1)
        weight=w/(1+(residual/.01)**2);weight/=max(weight.sum(),1e-12)
        mx=(x*weight[:,None]).sum(0);my=(y*weight[:,None]).sum(0)
        u,_,vt=np.linalg.svd((x-mx).T@((y-my)*weight[:,None]))
        sign=np.eye(3);sign[2,2]=np.linalg.det(vt.T@u.T)
        rotation=vt.T@sign@u.T
        pose[:3,:3]=rotation;pose[:3,3]=my-rotation@mx
    residual=np.linalg.norm(x@pose[:3,:3].T+pose[:3,3]-y,axis=-1)
    inliers=residual<.01;ratio=float(inliers.mean())
    receipt.update(inliers=int(inliers.sum()),inlier_ratio=ratio,residual_median_mm=float(np.median(residual)*1000))
    if inliers.sum()<12 or ratio<.2 or np.median(residual)>.02 or pose[2,3]<=0:return base.copy(),receipt
    receipt['accepted']=True
    return pose,receipt
