"""Read-only endpoint and PnP diagnostics on the frozen probe protocol."""
import numpy as np
import torch
from lip.unified.cad_image_correspondence import image_correspondence_targets,solve_correspondences,image_grid


@torch.no_grad()
def diagnose(output,target,scene,observation,visible,truth,mesh,seed):
    cpu=lambda x:x.detach().float().cpu().numpy()
    d=scene.diameter
    result={}
    if 'cad_image_uv' in output:
        labels=image_correspondence_targets(output,target,scene.k_crop[None],observation.diameter,visible)
        error=(output['cad_image_uv']-labels['uv']).norm(dim=-1)
        result['endpoints']={}
        for name in ('observed','real','proxy'):
            mask=labels[name]
            result['endpoints'][name]=dict(count=int(mask.sum()),epe_px=float(error[mask].mean()),within3px=float((error[mask]<3).float().mean())) if mask.any() else None
        for name in ('support','visible'):
            label=labels[name];known=labels['known_'+name]
            prediction=output['cad_image_'+name+'_logits'].sigmoid()>=.5
            result[name]=dict(known=int(known.sum()),positives=int((label&known).sum()),
                precision=float((prediction&label&known).sum()/(prediction&known).sum().clamp_min(1)),
                recall=float((prediction&label&known).sum()/(label&known).sum().clamp_min(1)))
    k=cpu(scene.k_crop);base=cpu(scene.pose)
    cases={}
    if 'cad_image_uv' in output:
        cases['cad_to_image']=(cpu(output['cad_image_xyz'][0])*d,cpu(output['cad_image_uv'][0]),cpu(output['cad_image_confidence'][0]))
    # Explicitly expose the existing inverse correspondences. Image coordinates
    # already identify each dense output; they were never absent from the map.
    h,w=output['surface_xyz'].shape[-2:]
    uv=image_grid(h,w,1,scene.pose.device)
    xyz=output['surface_xyz'][0].flatten(1).T
    support=output['support_logits'].sigmoid().reshape(1,1,16,16).repeat_interleave(14,-2).repeat_interleave(14,-1)
    conf=(output['geometry_valid_logits'].sigmoid()*support*scene.bounds).flatten()
    # One point per7x7 image cell: retain spatial coverage without GT masks.
    ids=(torch.arange(3,h,7,device=xyz.device)[:,None]*w+torch.arange(3,w,7,device=xyz.device)[None]).flatten()
    cases['image_to_cad']=(cpu(xyz[ids])*d,cpu(uv[ids]),cpu(conf[ids]))
    poses={'base':base,'frozen_head':cpu(output['pose_centered'][0])}
    result['solvers']={}
    for name,(points,pixels,weights) in cases.items():
        poses[name],result['solvers'][name]=solve_correspondences(points,pixels,weights,k,base,seed%2147483647)
    gt=cpu(truth);points=np.asarray(mesh['points'],dtype=np.float64)
    if len(points)>2048:points=points[np.linspace(0,len(points)-1,2048).astype(int)]
    reference=points@gt[:3,:3].T+gt[:3,3]
    from scipy.spatial import cKDTree
    tree=cKDTree(reference)
    result['pose']={}
    for name,pose in poses.items():
        predicted=points@pose[:3,:3].T+pose[:3,3]
        add=float(np.linalg.norm(predicted-reference,axis=-1).mean())
        adds=float(tree.query(predicted,k=1)[0].mean())
        angle=np.arccos(np.clip((np.trace(pose[:3,:3]@gt[:3,:3].T)-1)/2,-1,1))*180/np.pi
        result['pose'][name]=dict(rotation_deg=float(angle),translation_mm=float(np.linalg.norm(pose[:3,3]-gt[:3,3])*1000),
            add_mm=add*1000,adds_mm=adds*1000,adds_005d=adds<.05*d,add_01d=add<.1*d)
    return result
