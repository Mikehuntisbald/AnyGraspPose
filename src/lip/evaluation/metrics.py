import numpy as np
import torch
from scipy.spatial import cKDTree
from lip.geometry.so3 import angle, log, exp


def constant_velocity(history, times, target_time):
    base=history[-1].clone()
    if len(history)<2 or float(times[-1]-times[-2])<=0:return base
    ratio=(target_time-times[-1])/(times[-1]-times[-2])
    base[:3, :3]=exp(log(history[-1, :3, :3] @ history[-2, :3, :3].T)*ratio) @ history[-1, :3, :3]
    base[:3, 3]+=ratio*(history[-1, :3, 3]-history[-2, :3, 3])
    return base


def errors(pred, gt, points, d):
    pred,gt,points=[np.asarray(x,dtype='f8') for x in (pred,gt,points)]
    a=points@pred[:3,:3].T+pred[:3,3];b=points@gt[:3,:3].T+gt[:3,3]
    add=float(np.linalg.norm(a-b,axis=1).mean());adds=float(cKDTree(b).query(a)[0].mean())
    return dict(center_mm=float(np.linalg.norm(pred[:3,3]-gt[:3,3])*1000),
                rotation_deg=float(angle(torch.from_numpy(pred[:3,:3]@gt[:3,:3].T))*180/torch.pi),
                add_m=add,adds_m=adds,add_005=float(add<.05*d),add_01=float(add<.1*d),
                adds_005=float(adds<.05*d),adds_01=float(adds<.1*d))


def summarize(rows):
    if not rows:return {'count':0}
    rows=[dict(r,physical_sequence='/'.join(r['stream_id'].split('/')[:2])) for r in rows]
    keys=['center_mm','rotation_deg','add_m','adds_m','add_005','add_01','adds_005','adds_01','lost']
    def stats(rs):
        ans={'count':len(rs)}
        for k in keys:
            a=np.array([r[k] for r in rs if k in r],dtype=float)
            if len(a):ans[k]={'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95))}
        return ans
    output={'micro':stats(rows)}
    for group in ['object_id','stream_id','physical_sequence','visibility_bin','moving']:
        groups={}
        for r in rows:groups.setdefault(str(r[group]),[]).append(r)
        output['per_'+group]={k:stats(v) for k,v in groups.items()}
    output['macro_object']={k:float(np.mean([v[k]['mean'] for v in output['per_object_id'].values()])) for k in keys if k in output['micro']}
    output['macro_sequence']={k:float(np.mean([v[k]['mean'] for v in output['per_stream_id'].values()])) for k in keys if k in output['micro']}
    output['macro_physical_sequence']={k:float(np.mean([v[k]['mean'] for v in output['per_physical_sequence'].values()])) for k in keys if k in output['micro']}
    return output


def visibility_bin(v):
    if v is None:return 'unknown'
    return '[0,.1)' if v<.1 else '[.1,.3)' if v<.3 else '[.3,.6)' if v<.6 else '[.6,1]'
