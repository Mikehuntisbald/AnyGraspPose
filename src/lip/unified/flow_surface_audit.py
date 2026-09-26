"""Frozen information audit: can predicted endpoints recover CAD geometry?

Not an inference architecture, training objective, or pose-head change. Candidate
surface construction has no teacher arguments; oracle controls are built later.
"""
import numpy as np
import torch


def triangulated_transport(xyz, endpoints, available, reference_depth, side=224):
    """Transport structured template triangles; never average separate sheets.

Affine barycentric interpolation is a diagnostic approximation. Full coverage
and perspective accuracy must be measured with the oracle-endpoint control.
Reject folds and large canonical discontinuities. Resolve overlaps by estimated
template depth (not GT depth). Uncovered pixels are explicit, not zero targets.
"""
    xyz=np.asarray(xyz,dtype=np.float64).reshape(16,16,3)
    uv=np.asarray(endpoints,dtype=np.float64).reshape(16,16,2)
    good=np.asarray(available,dtype=bool).reshape(16,16)
    z=np.asarray(reference_depth,dtype=np.float64).reshape(16,16)
    output=np.zeros((side,side,3),dtype=np.float32)
    zbuf=np.full((side,side),np.inf);covered=np.zeros((side,side),dtype=bool)
    triangles=folds=discontinuities=0
    for y in range(15):
        for x in range(15):
            for ids in (((y,x),(y,x+1),(y+1,x)),((y+1,x+1),(y+1,x),(y,x+1))):
                rows,cols=np.array(ids).T
                if not good[rows,cols].all():continue
                points=xyz[rows,cols];p=uv[rows,cols];depth=z[rows,cols]
                if not np.isfinite(p).all() or not np.isfinite(points).all():continue
                edges=np.linalg.norm(points[[1,2,0]]-points,axis=1)
                if edges.max()>.2:discontinuities+=1;continue
                matrix=(p[1:]-p[0]).T
                determinant=np.linalg.det(matrix)
                if determinant<1.:folds+=1;continue
                xmin,ymin=np.maximum(np.floor(p.min(0)).astype(int),0)
                xmax,ymax=np.minimum(np.ceil(p.max(0)).astype(int),side-1)
                if xmin>xmax or ymin>ymax:continue
                yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1]
                xy=np.stack((xx,yy),-1)
                ab=(xy-p[0])@np.linalg.inv(matrix).T
                weights=np.concatenate((1-ab.sum(-1,keepdims=True),ab),-1)
                inside=(weights>=-1e-8).all(-1)
                interpolated_z=weights@depth
                tile=zbuf[ymin:ymax+1,xmin:xmax+1]
                take=inside&(interpolated_z<tile)
                if not take.any():continue
                tile[take]=interpolated_z[take]
                output[ymin:ymax+1,xmin:xmax+1][take]=(weights@points)[take]
                covered[ymin:ymax+1,xmin:xmax+1][take]=True
                triangles+=1
    return output,covered,dict(triangles=triangles,rejected_folds=folds,rejected_discontinuities=discontinuities)


def fit_rigid_surface(xyz, uv, confidence, crop_k, base, seed):
    from .cad_image_correspondence import solve_correspondences
    # Same existing audited solver, no GT-based acceptance or candidate ranking.
    return solve_correspondences(xyz,uv,confidence,crop_k,base,seed)


def fit_metric_surface(xyz,uv,depth,confidence,crop_k,base):
    from .cad_image_correspondence import solve_rgbd_correspondences
    ray=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(crop_k).T
    camera=ray*np.asarray(depth)[:,None]
    return solve_rgbd_correspondences(xyz,camera,confidence,base)


def fuse_depth_evidence(recovered,measured,visible,inside):
    """Continuous measured support; keep inferred and measured mass explicit."""
    valid=np.isfinite(measured)&(measured>0)&inside
    safe=np.where(valid,measured,recovered)
    weight=np.asarray(visible)*valid*np.exp(-.5*((safe-recovered)/.015)**2)
    mass=.25+weight
    return (.25*recovered+weight*safe)/mass,mass,weight/mass


@torch.no_grad()
def diagnose_flow_surface(output, target, scene, observation, visible, renderer, seed):
    from .flow_reconstruction import flow_labels
    ref=output['flow_reference'];flow=output['flow_rounds'][-1]
    cpu=lambda v:v.detach().float().cpu().numpy()
    xyz=cpu(ref['xyz'][0]);uv=cpu(flow['uv'][0]);available=ref['available'][0].cpu().numpy()
    source_z=cpu(ref['depth'][0]);confidence=cpu(flow['support_logits'][0].sigmoid())*available
    confidence=np.where(confidence>=.5,confidence,0.)
    d=float(observation.diameter[0]);k=cpu(scene.k_crop);base=cpu(observation.base[0])
    original_xyz=output['surface_xyz'];original_depth=output['surface_depth_m']
    def tensor(array):return torch.as_tensor(array,device=original_xyz.device)
    def triangle(points,endpoints,keep):
        value,mask,receipt=triangulated_transport(points,endpoints,keep,source_z)
        value=tensor(value).permute(2,0,1)[None];mask=tensor(mask)[None,None].bool()
        return dict(xyz=torch.where(mask,value,original_xyz),depth=original_depth,covered=mask,receipt=receipt)
    def rendered(pose,receipt):
        if receipt['accepted']:
            render=renderer(scene.cad['appearance'],tensor(pose).float(),scene.k_crop,224)
            mask=render['mask'][None,None].bool()
            return dict(xyz=torch.where(mask,render['xyz'][None]/d,original_xyz),
                        depth=torch.where(mask,render['depth'][None],original_depth),covered=mask,receipt=receipt)
        return dict(xyz=original_xyz,depth=original_depth,covered=torch.zeros_like(original_depth,dtype=torch.bool),receipt=receipt)
    def rigid(points,endpoints,weights):
        return rendered(*fit_rigid_surface(points*d,endpoints,weights,k,base,seed))
    # Prediction-only branches constructed before consulting any target values.
    candidates=dict(triangle=triangle(xyz,uv,available),rigid=rigid(xyz,uv,confidence))
    from .cad_image_correspondence import sample_points
    sampled_depth=cpu(sample_points(original_depth,flow['uv'])[0,:,0])
    measured=cpu(sample_points(observation.measured_depth_m,flow['uv'],mode='nearest')[0,:,0])
    predicted_visible=cpu(flow['visible_logits'][0].sigmoid())
    inside=(uv>=0).all(-1)&(uv<224).all(-1)
    metric_confidence=confidence*inside*(sampled_depth>0)*np.isfinite(sampled_depth)
    candidates['metric_recovered']=rendered(*fit_metric_surface(xyz*d,uv,sampled_depth,metric_confidence,k,base))
    # evidence_logits is a PATCH AREA FRACTION, not point visibility. Earlier
    # hard visibility/depth guards discarded all measured anchors in this audit.
    # Retain soft point visibility,
    # attenuate depth disagreement and expose actual measured contribution.
    mixed,mass,measured_fraction=fuse_depth_evidence(sampled_depth,measured,predicted_visible,inside)
    weights=metric_confidence*mass
    candidates['metric_measured']=rendered(*fit_metric_surface(xyz*d,uv,mixed,weights,k,base))
    candidates['metric_measured']['receipt']['measured_anchors']=int(((measured_fraction>.01)&(metric_confidence>0)).sum())
    candidates['metric_measured']['receipt']['measured_fraction_mean']=float(measured_fraction[metric_confidence>0].mean()) if (metric_confidence>0).any() else 0.
    labels=flow_labels(output,target,scene.k_crop[None],observation.diameter,visible)
    oracle_keep=available&labels['support'][0].cpu().numpy()
    oracle_uv=cpu(labels['uv'][0])
    candidates['oracle_triangle']=triangle(xyz,oracle_uv,oracle_keep)
    candidates['oracle_rigid']=rigid(xyz,oracle_uv,oracle_keep.astype(float))
    candidates['oracle_cad']=dict(xyz=target.cad_geometry_xyz,depth=target.cad_geometry_depth_m,
        covered=target.cad_geometry_valid,receipt=dict(teacher_only=True,solver=False))
    def score(value,domain):
        if not domain.any():return None
        canonical=domain&target.cad_geometry_valid
        xyz_error=(value['xyz']-target.cad_geometry_xyz).norm(dim=1,keepdim=True)*d*1000
        depth_error=(value['depth']-target.surface_depth_m).abs()*1000
        covered=domain&value['covered']
        covered_xyz=canonical&value['covered']
        return dict(pixels=int(domain.sum()),coverage=float(covered.sum()/domain.sum()),
            canonical_xyz_mm=float(xyz_error[canonical].mean()) if canonical.any() else None,
            depth_mm=float(depth_error[domain].mean()),
            covered_canonical_xyz_mm=float(xyz_error[covered_xyz].mean()) if covered_xyz.any() else None,
            covered_depth_mm=float(depth_error[covered].mean()) if covered.any() else None)
    result={name:dict(receipt=v['receipt'],real=score(v,target.geometry_real_weight),proxy=score(v,target.geometry_proxy_weight))
            for name,v in candidates.items()}
    result['point_counts']=dict(available=int(available.sum()),predicted_support=int((confidence>0).sum()),
        oracle_support=int(oracle_keep.sum()),point_visible_above_half=int(((predicted_visible>=.5)&(confidence>0)).sum()),
        measured_depth_available=int(((measured>0)&np.isfinite(measured)&inside&(confidence>0)).sum()),
        point_visibility_max=float(predicted_visible[confidence>0].max()) if (confidence>0).any() else None)
    known=labels['known_visible'][0].cpu().numpy()
    result['visibility_samples']=dict(score=predicted_visible[known].tolist(),
        label=labels['visible'][0].cpu().numpy()[known].astype(int).tolist())
    return result,candidates
