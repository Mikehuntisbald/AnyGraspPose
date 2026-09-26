"""Read-only endpoint and PnP diagnostics on the frozen probe protocol."""
import numpy as np
import torch
from lip.unified.cad_image_correspondence import image_correspondence_targets,solve_correspondences,image_grid,solve_rgbd_correspondences,sample_points


@torch.no_grad()
def diagnose(output,target,scene,observation,visible,truth,mesh,seed,renderer=None,model=None):
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
        selected=(output['cad_image_support_logits'][0]>=0)&(output['cad_image_visible_logits'][0]>=0)
        cases['cad_to_image_visible']=(cases['cad_to_image'][0],cases['cad_to_image'][1],cpu(output['cad_image_confidence'][0]*selected))
        if 'cad_flow_uv' in output:
            cases['anchored_flow']=(cases['cad_to_image'][0],cpu(output['cad_flow_uv'][0]),cpu(output['cad_image_confidence'][0]*selected))
            error=(output['cad_flow_uv']-labels['uv']).norm(dim=-1)
            result['flow_endpoints']={n:float(error[labels[n]].mean()) if labels[n].any() else None for n in ('observed','real','proxy')}
        if model is not None:
            result['search_prior_endpoints']={}
            for sigma in (16.,32.):
                alternative=model.cad_atlas_decoder.image_readout(output,output['surface_xyz'].shape[-2:],
                    estimated_uv=observation.cad_atlas[1][:,:,15:17]*224,prior_sigma=sigma)
                tag='cad_to_image_prior'+str(int(sigma))
                alt_uv=alternative['cad_image_uv'][0]
                selected=(alternative['cad_image_support_logits'][0]>=0)&(alternative['cad_image_visible_logits'][0]>=0)
                cases[tag]=(cpu(alternative['cad_image_xyz'][0])*d,cpu(alt_uv),cpu(alternative['cad_image_confidence'][0]*selected))
                cases[tag+'_all']=(cases[tag][0],cases[tag][1],cpu(alternative['cad_image_confidence'][0]))
                error=(alternative['cad_image_uv']-labels['uv']).norm(dim=-1)
                result['search_prior_endpoints'][tag]={n:float(error[labels[n]].mean()) if labels[n].any() else None for n in ('observed','real','proxy')}
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
    evidence=output['evidence_logits'].sigmoid().reshape(1,1,16,16).repeat_interleave(14,-2).repeat_interleave(14,-1)
    mask=((evidence>=.7)&(output['geometry_valid_logits'].sigmoid()>=.5)&(support>=.5)).flatten()
    cases['image_to_cad_visible']=(cpu(xyz[ids])*d,cpu(uv[ids]),cpu(conf[ids]*mask[ids]))
    poses={'base':base,'frozen_head':cpu(output['pose_centered'][0])}
    result['solvers']={}
    for name,(points,pixels,weights) in cases.items():
        poses[name],result['solvers'][name]=solve_correspondences(points,pixels,weights,k,base,seed%2147483647)
    # RGB-D extension of correspondence solving: consume the decoded depth,
    # retain trustworthy current measurements, and cap total inferred mass.
    for name in ('image_to_cad','cad_to_image','anchored_flow'):
        if name not in cases:continue
        points,pixels,weights=cases[name]
        pixel_tensor=torch.tensor(pixels,device=scene.pose.device,dtype=torch.float32)[None]
        measured=cpu(sample_points(observation.measured_depth_m,pixel_tensor,'nearest')[0,:,0])
        recovered=cpu(sample_points(output['surface_depth_m'],pixel_tensor)[0,:,0])
        valid=cpu(sample_points(output['geometry_valid_logits'].sigmoid(),pixel_tensor)[0,:,0])>=.5
        vis=cpu(sample_points(evidence,pixel_tensor)[0,:,0])>=.7
        genuine=vis&(measured>0)&np.isfinite(measured)&(np.abs(measured-recovered)<.03)
        measured_w=weights*genuine
        completed_w=weights*(~genuine)*valid*.2
        if measured_w.sum()>0:completed_w*=min(1.,measured_w.sum()/max(completed_w.sum(),1e-12))
        rays=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(k).T
        for kind,z,wgt in [('measured',measured,measured_w),('recovered',recovered,weights*valid*.2),
                           ('mixed',np.where(genuine,measured,recovered),measured_w+completed_w)]:
            tag=name+'_rgbd_'+kind
            poses[tag],result['solvers'][tag]=solve_rgbd_correspondences(points,rays*z[:,None],wgt,base)
            result['solvers'][tag].update(measured_mass=float(measured_w.sum()),completed_mass=float(completed_w.sum()))
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
    if renderer is not None:
        result['rigid_geometry']={}
        for name in ('base','image_to_cad_visible','cad_to_image_visible','cad_to_image_prior16','cad_to_image_prior32',
                     'image_to_cad_rgbd_mixed','cad_to_image_rgbd_mixed','anchored_flow','anchored_flow_rgbd_mixed'):
            if name not in poses:continue
            render=renderer(scene.cad['appearance'],torch.as_tensor(poses[name],device=scene.pose.device,dtype=torch.float32),scene.k_crop,224)
            covered=render['mask'][None,None]&scene.bounds
            # Fixed fallback policy for every arm; missing predicted silhouettes
            # cannot silently remove difficult evaluation pixels.
            xyz=torch.where(covered,render['xyz'][None]/d,output['surface_xyz'])
            depth=torch.where(covered,render['depth'][None],output['surface_depth_m'])
            dx=(xyz-target.cad_geometry_xyz).norm(dim=1,keepdim=True)*d*1000
            dz=(depth-target.surface_depth_m).abs()*1000
            regions={}
            for kind,mask in [('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight)]:
                regions[kind]=dict(pixels=int(mask.sum()),xyz_mm=float(dx[mask].mean()),depth_mm=float(dz[mask].mean()),
                    covered_fraction=float(covered[mask].float().mean()),
                    all_target_good_10mm_xyz_5mm_depth=float(((dx<10)&(dz<5)&covered)[mask].float().mean())) if mask.any() else None
            result['rigid_geometry'][name]=regions
    return result
