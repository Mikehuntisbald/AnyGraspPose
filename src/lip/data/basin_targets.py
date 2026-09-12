"""Lossless before/after FP outcomes and explicit, re-derivable target policy."""
import numpy as np
import torch
from lip.evaluation.metrics import errors
from lip.geometry.so3 import original_pose
SCHEMA_VERSION=2

def label_policy(metric='adds_over_d',tau=.1,margin=.005):
    if metric not in ('add_over_d','adds_over_d'):raise ValueError('Unsupported primary error')
    if not np.isfinite(tau) or tau<=0 or not np.isfinite(margin) or margin<0:raise ValueError('Invalid target thresholds')
    return dict(primary_error=metric,tau=float(tau),improve_margin=float(margin),
                converge_rule='e_after < tau',improve_rule='e_after < e_before - margin',
                delta_rule='e_before - e_after; positive means improvement',
                pose_frame='object mesh-centered to camera; original-mesh aliases also stored',
                center_error='Euclidean error of mesh center in camera coordinates, millimeters',
                rotation_error='canonical SO(3) geodesic angle, degrees; no symmetry minimization')

def fields_from_metrics(before,after,diameter,metric='adds_over_d',tau=.1,margin=.005):
    policy=label_policy(metric,tau,margin)
    if not np.isfinite(diameter) or diameter<=0:raise ValueError('Invalid mesh diameter')
    if not len(before) or len(before)!=len(after):raise ValueError('Candidate count mismatch')
    result={}
    for side,rows in [('before',before),('after',after)]:
        for name,key,scale in [('add','add_m',diameter),('adds','adds_m',diameter),('rotation','rotation_deg',1.),('center','center_mm',1.)]:
            suffix='over_d' if name in ('add','adds') else 'deg' if name=='rotation' else 'mm'
            values=np.array([r[key] for r in rows],dtype=np.float64)/scale
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid continuous error')
            result[f'{name}_{side}_{suffix}']=values
    for name,suffix in [('add','over_d'),('adds','over_d'),('rotation','deg'),('center','mm')]:
        result[f'delta_{name}_{suffix}']=result[f'{name}_before_{suffix}']-result[f'{name}_after_{suffix}']
    primary='adds' if metric=='adds_over_d' else 'add'
    result['e_before']=result[f'{primary}_before_over_d'].copy();result['e_after']=result[f'{primary}_after_over_d'].copy()
    result['delta_e']=result['e_before']-result['e_after']
    result['y_converge']=(result['e_after']<tau).astype(np.uint8)
    result['y_improve']=(result['e_after']<result['e_before']-margin).astype(np.uint8)
    result['diameter_m']=np.array(diameter,dtype=np.float64)
    result['target_tau']=np.array(tau,dtype=np.float64);result['target_margin']=np.array(margin,dtype=np.float64)
    result['target_metric']=np.array(metric);result['targets_schema_version']=np.array(SCHEMA_VERSION,dtype=np.int32)
    return result

def candidate_outcomes(candidates,refined,gt,vertices,center,diameter,after_metrics=None,metric='adds_over_d',tau=.1,margin=.005):
    candidates=np.asarray(candidates);refined=np.asarray(refined);gt=np.asarray(gt)
    if candidates.ndim!=3 or candidates.shape[1:]!=(4,4) or refined.shape!=candidates.shape:raise ValueError('Expected N candidate/refined 4x4 poses')
    if not np.isfinite(candidates).all() or not np.isfinite(refined).all():raise ValueError('Nonfinite FP outcome requires an explicit failure record')
    before=[errors(t,gt,vertices,diameter) for t in candidates]
    after=after_metrics if after_metrics is not None else [errors(t,gt,vertices,diameter) for t in refined]
    result=fields_from_metrics(before,after,diameter,metric,tau,margin)
    result.update(candidate_pose=candidates.copy(),refined_pose=refined.copy(),mesh_center_m=np.array(center,dtype=np.float64),
                  candidate_pose_original=original_pose(torch.from_numpy(candidates),torch.as_tensor(center,dtype=torch.from_numpy(candidates).dtype)).numpy(),
                  refined_pose_original=original_pose(torch.from_numpy(refined),torch.as_tensor(center,dtype=torch.from_numpy(refined).dtype)).numpy())
    return result
