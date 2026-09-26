"""Recovery metrics in explicit coordinate frames; targets never gate predictions."""
import torch
from .surface_normals import normal_terms


@torch.no_grad()
def recovery_metrics(output,target,diameter,crop_k,masks,visible_depth_m=None):
    """Return one dictionary per batch lane, with fixed real/proxy/visible masks.

    Means/percentiles are per frame. Canonical XYZ is CAD identity; physical
    camera XYZ is predicted depth lifted through calibrated pixel rays.
    """
    xyz=output['surface_xyz'].float();depth=output['surface_depth_m'].float()
    probability=output['geometry_valid_logits'].float().sigmoid()
    if not all(torch.isfinite(x).all() for x in (xyz,depth,probability)):
        raise ValueError('Nonfinite recovery output; do not silently drop failed predictions')
    d=diameter[:,None,None,None].float()
    cad=(xyz-target.cad_geometry_xyz).norm(dim=1,keepdim=True)*d*1000
    camera=target.camera_rays*depth
    identity_camera=(torch.einsum('bij,bjhw->bihw',target.camera_rotation,xyz)+target.camera_translation_d[:,:,None,None])*d
    projected=torch.einsum('bij,bjhw->bihw',crop_k.float(),identity_camera)
    h,w=depth.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device=depth.device),torch.arange(w,device=depth.device),indexing='ij')
    uv=torch.stack((xx,yy),0)[None]
    endpoint=(projected[:,:2]/projected[:,2:3].clamp_min(.001)-uv).norm(dim=1,keepdim=True)
    rows=[{} for _ in range(len(depth))]
    for name,mask in masks.items():
        if name=='visible' and visible_depth_m is None:raise ValueError('Visible evaluation requires original sensor depth')
        truth_depth=visible_depth_m if name=='visible' else target.surface_depth_m
        dep=(depth-truth_depth).abs()*1000
        camera_truth=target.camera_rays*truth_depth
        camera_error=(camera-camera_truth).norm(dim=1,keepdim=True)*1000
        mask=mask.bool()
        _,normal_valid,cosine=normal_terms(camera/d,camera_truth/d,mask,2)
        angle=cosine.acos()*180/torch.pi
        for lane in range(len(depth)):
            active=mask[lane];count=int(active.sum())
            if not count:rows[lane][name]=None;continue
            values={key:tensor[lane][active] for key,tensor in [('canonical_xyz_mm',cad),('depth_mm',dep),('camera_xyz_mm',camera_error),('identity_reprojection_px',endpoint)]}
            names=list(values);packed=[v.mean() for v in values.values()]
            for key in ('canonical_xyz_mm','depth_mm'):
                names += [key.replace('_mm','_median_mm'),key.replace('_mm','_p90_mm')]
                packed += [values[key].median(),values[key].quantile(.9)]
            names += ['canonical_within10mm','depth_within5mm','valid_recall','joint_xyz10mm_depth5mm','identity_behind_camera_fraction']
            packed += [(cad[lane][active]<10).float().mean(),(dep[lane][active]<5).float().mean(),
                (probability[lane][active]>=.5).float().mean(),
                ((cad[lane][active]<10)&(dep[lane][active]<5)&(probability[lane][active]>=.5)).float().mean(),
                (identity_camera[lane,2:3][active]<=0).float().mean()]
            nv=normal_valid[lane];normal_count=int(nv.sum())
            result=dict(pixels=count,**dict(zip(names,torch.stack(packed).cpu().tolist())),normal_stencils=normal_count,
                camera_normal_deg=float(angle[lane][nv].mean()) if normal_count else None)
            rows[lane][name]=result
    label=target.geometry_valid_label
    for lane in range(len(rows)):
        known=torch.isfinite(label[lane]);positive=known&(label[lane]>=.5);negative=known&(label[lane]<.5)
        rows[lane]['validity']=dict(known_pixels=int(known.sum()),positive_pixels=int(positive.sum()),negative_pixels=int(negative.sum()),
            brier=float((probability[lane][known]-label[lane][known]).square().mean()) if known.any() else None,
            tpr=float((probability[lane][positive]>=.5).float().mean()) if positive.any() else None,
            fpr=float((probability[lane][negative]>=.5).float().mean()) if negative.any() else None)
    return rows


@torch.no_grad()
def visible_eval_mask(target,visible,added,raw_depth,bounds):
    """Visible depth is evaluated against real sensor depth, never CAD depth."""
    import torch.nn.functional as F
    mask=visible&~added&bounds&torch.isfinite(raw_depth)&(raw_depth>0)&target.cad_geometry_valid
    return -F.max_pool2d(-mask.float(),5,1,2)>.999
