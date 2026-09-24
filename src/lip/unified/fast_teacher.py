"""Batch image-space target construction, keeping per-view matrix arithmetic."""
import torch
from torch.nn import functional as F


@torch.no_grad()
@torch.autocast('cuda',enabled=False)
def build_fast_teacher(encoder,scenes,gt_poses,visible_masks,added_masks,renderer,max_radius_d,batch_render):
    from .features import TeacherTargets,valid_real_geometry,normalize_rgb
    from .execution_speed import crop_images_fast,camera_points_fast
    renders=renderer.render_many([s.cad['appearance'] for s in scenes],gt_poses,[s.k_crop for s in scenes],224) if batch_render else [renderer(s.cad['appearance'],pose.float(),s.k_crop,224) for s,pose in zip(scenes,gt_poses)]
    visible=[];real_xyz=[];rays=[]
    for s,pose,mask in zip(scenes,gt_poses,visible_masks):
        visible.append(crop_images_fast(mask[None].float(),s.affine,mode='nearest')>.5)
        ray=camera_points_fast(torch.ones_like(s.depth),s.k_crop)
        xyz=((ray*s.depth[0,0,:,:,None]-pose[:3,3])@pose[:3,:3]/s.diameter).permute(2,0,1)[None]
        real_xyz.append(xyz);rays.append(ray.permute(2,0,1))
    v=torch.cat(visible);real_xyz=torch.cat(real_xyz);depth=torch.cat([s.depth for s in scenes])
    bounds=torch.cat([s.bounds for s in scenes]);added=torch.cat(added_masks)
    silhouette=torch.stack([r['mask'] for r in renders])[:,None]
    rendered_depth=torch.stack([r['depth'] for r in renders])
    # Preserve Python-scalar division rounding from the original per-view path.
    rendered_xyz=torch.cat([r['xyz'][None]/s.diameter for r,s in zip(renders,scenes)])
    original=torch.cat([s.rgb for s in scenes]);rgb=torch.stack([r['rgb'] for r in renders])
    proxy=torch.where(silhouette,rgb,original);hidden=silhouette&~v
    erode=lambda m:-F.max_pool2d(-m.float(),5,1,2)>.999
    good=erode(silhouette)&bounds&(rendered_depth>0)
    known=erode(v)&silhouette&bounds;hidden_good=erode(hidden)&bounds
    artificial=v&added;artificial_interior=erode(artificial)&known
    real_valid,rejected=valid_real_geometry(depth,real_xyz,max_radius_d)
    real_good=artificial_interior&erode(real_valid);proxy_good=good&hidden_good
    target_xyz=torch.where(artificial,real_xyz,rendered_xyz)
    target_depth=torch.where(artificial,depth,rendered_depth)
    residual=torch.cat([(target_depth[i:i+1]-s.pose[2,3])/s.diameter for i,s in enumerate(scenes)])
    label=(silhouette&(rendered_depth>0)).float()
    label=torch.where(artificial,real_valid.float(),label)
    label=label.masked_fill(artificial&rejected,float('nan'))
    label=label.masked_fill(~bounds,float('nan')).masked_fill(v&~silhouette,float('nan')).masked_fill(v&~added,float('nan'))
    boundary=~erode(silhouette)&~erode(~silhouette);label=label.masked_fill(boundary,float('nan'))
    with torch.autocast(original.device.type,dtype=torch.bfloat16):mid,last=encoder(normalize_rgb(torch.cat((original,proxy))))
    b=len(scenes);real_mid,real_last,proxy_mid,proxy_last=mid[:b].float(),last[:b].float(),mid[b:].float(),last[b:].float()
    pool=lambda x:F.avg_pool2d(x.float(),14,14).flatten(1)
    patch_bounds=pool(bounds)>=.999;known_patch=pool(known)>=.9;complete_proxy=pool(hidden_good)>=.9
    real_hidden=pool(artificial_interior)>=.9;added_patch=pool(added)
    visible_label=pool(v&~added).masked_fill(~patch_bounds,float('nan'))
    support_label=pool(silhouette).masked_fill(~patch_bounds,float('nan'))
    return TeacherTargets(real_mid,real_last,proxy_mid,proxy_last,
        known_patch*patch_bounds*(1-added_patch),real_hidden*patch_bounds,complete_proxy*patch_bounds,
        complete_proxy*known_patch*patch_bounds,complete_proxy*complete_proxy*patch_bounds,
        visible_label,support_label,proxy,original,target_xyz,residual,target_depth,
        real_good|proxy_good,real_good,proxy_good,label,real_good,proxy_good,
        torch.stack([pose[:3,:3].float() for pose in gt_poses]),
        torch.stack([pose[:3,3].float()/s.diameter for pose,s in zip(gt_poses,scenes)]),
        torch.stack(rays),torch.stack([s.pose[2,3]/s.diameter for s in scenes]))
