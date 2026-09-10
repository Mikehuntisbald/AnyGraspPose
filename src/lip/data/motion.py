import json
from pathlib import Path
import numpy as np
import torch
from lip.geometry.so3 import center_pose,angle


def motion_thresholds(index_root,quantile=.75):
    """Freeze speed cutoffs using only train transitions, never val/test metrics."""
    index=Path(index_root);path=index/'motion_thresholds.json'
    audit=json.loads((index/'audit.json').read_text())
    if path.exists():
        result=json.loads(path.read_text())
        if result['split_hash']!=audit['split_hash'] or result['quantile']!=quantile:raise ValueError('Stale train motion thresholds')
        return result
    trans=[];rot=[]
    for line in (index/'streams.jsonl').read_text().splitlines():
        s=json.loads(line)
        if s['split']!='train':continue
        with np.load(index/s['mesh_cache']) as m:center=torch.from_numpy(m['center'].copy());d=float(m['diameter'])
        with np.load(index/s['pose_cache']) as p:
            poses=center_pose(torch.from_numpy(p['poses']),center);dt=torch.from_numpy(np.diff(p['frames']).astype('f4'))/audit['fps']
        if len(poses)<2:continue
        trans.extend(((poses[1:,:3,3]-poses[:-1,:3,3]).norm(dim=-1)/d/dt).tolist())
        rot.extend((angle(poses[1:,:3,:3]@poses[:-1,:3,:3].transpose(-1,-2))/dt).tolist())
    if not trans:raise ValueError('No training motion transitions available')
    result=dict(source='train transitions only',split_hash=audit['split_hash'],quantile=quantile,transitions=len(trans),
                center_d_per_sec=float(np.quantile(trans,quantile)),rotation_rad_per_sec=float(np.quantile(rot,quantile)))
    path.write_text(json.dumps(result,indent=2));return result
