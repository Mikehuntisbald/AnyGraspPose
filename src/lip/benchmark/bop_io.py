"""DexYCB-original BOP CSV convention: R row-major, t in millimeters."""
import csv
from pathlib import Path
import numpy as np

FIELDS=['scene_id','im_id','obj_id','score','R','t','time']

def load_predictions(path):
    rows=[]
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            t=np.eye(4,dtype=np.float64);t[:3,:3]=np.fromstring(r['R'],sep=' ').reshape(3,3)
            t[:3,3]=np.fromstring(r['t'],sep=' ')/1000.
            if not np.isfinite(t).all() or not np.isfinite(float(r['score'])):raise ValueError('Nonfinite initializer CSV')
            rows.append(dict(scene_id=int(r['scene_id']),im_id=int(r['im_id']),obj_id=int(r['obj_id']),score=float(r['score']),pose=t))
    return rows

def key(row):return row['scene_id'],row['im_id'],row['obj_id']

def select_initializers(rows,targets):
    """Use only the public VIVO target object/count and predicted confidence."""
    grouped={}
    for r in rows:grouped.setdefault(key(r),[]).append(r)
    selected=[];missing=[]
    for t in sorted(targets,key=key):
        candidates=sorted(grouped.get(key(t),[]),key=lambda r:r['score'],reverse=True)[:t['inst_count']]
        selected.extend(candidates)
        if len(candidates)<t['inst_count']:missing.append(dict(target=t,missing=t['inst_count']-len(candidates)))
    return selected,missing

def csv_row(row,pose):
    pose=np.asarray(pose,dtype=np.float64)
    if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('Invalid export pose')
    return dict(scene_id=row['scene_id'],im_id=row['im_id'],obj_id=row['obj_id'],score=row['score'],
        R=' '.join(format(float(x),'.17g') for x in pose[:3,:3].reshape(-1)),
        t=' '.join(format(float(x),'.17g') for x in pose[:3,3]*1000.),time=-1)
