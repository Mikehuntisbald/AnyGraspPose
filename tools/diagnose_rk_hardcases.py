"""Audit matched validation failures and export factual RGB/pose figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull,QhullError

ARMS=('R0K0','R1K0','R0K1','R1K1')


def read_evaluation(root):
    rows={};manifests={}
    for arm in ARMS:
        folder=root/arm/'s0_val';m=json.loads((folder/'manifest.json').read_text())
        if not m['completed'] or not m['population_verified']:raise ValueError('Completed full validation is required')
        manifests[arm]=m
        rows[arm]={(r['stream_id'],r['frame_index']):r for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines())}
    keys=set(rows[ARMS[0]])
    if any(set(r)!=keys for r in rows.values()):raise ValueError('Unpaired populations')
    return rows,manifests


def hull(points,pose,k):
    camera=points@pose[:3,:3].T+pose[:3,3]
    camera=camera[camera[:,2]>.01]
    projected=camera@k.T;xy=projected[:,:2]/projected[:,2:3]
    if len(xy)<3:return None
    try:boundary=xy[ConvexHull(xy).vertices]
    except QhullError:return None
    return np.concatenate((boundary,boundary[:1]))


def figure_case(key,rows,stream,data_root,index_root,out):
    sid,frame=key;image=np.asarray(Image.open(data_root/stream['relative_dir']/f'color_{frame:06d}.jpg').convert('RGB'))
    with np.load(index_root/stream['mesh_cache']) as z:points=z['vertices'].copy();center=z['center'].copy();diameter=float(z['diameter'])
    with np.load(index_root/stream['pose_cache']) as z:
        loc=np.flatnonzero(z['frames']==frame)
        if len(loc)!=1:raise ValueError('Frame identity mismatch')
        gt=z['poses'][loc[0]].copy()
    # Predictions use centered mesh coordinates, while cached vertices are centered.
    # Confirm their convention against the metadata used by errors/rendering.
    gt[:3,3]+=gt[:3,:3]@center
    k=np.asarray(stream['intrinsics']);gt_hull=hull(points,gt,k)
    fig,axes=plt.subplots(2,2,figsize=(12.8,9.6),dpi=130)
    for ax,arm in zip(axes.flat,ARMS):
        r=rows[arm][key];pred=np.asarray(r['pose_centered']);pred_hull=hull(points,pred,k)
        ax.imshow(image)
        if gt_hull is not None:ax.plot(gt_hull[:,0],gt_hull[:,1],color='lime',lw=1.7,label='GT projected hull')
        if pred_hull is not None:ax.plot(pred_hull[:,0],pred_hull[:,1],color='red',lw=1.3,label='Prediction hull')
        ax.set_xlim(0,image.shape[1]);ax.set_ylim(image.shape[0],0);ax.axis('off')
        ax.set_title(f"{arm}: ADD/d={r['add_m']/diameter:.3f}, ADD-S/d={r['adds_m']/diameter:.3f}\n"
            f"center={r['center_mm']:.1f} mm, rotation={r['rotation_deg']:.1f} deg, support={r.get('observation_support',float('nan')):.2f}")
    axes.flat[0].legend(loc='lower right',fontsize=8)
    fig.suptitle(f"{sid}  frame {frame}  visibility={rows[ARMS[0]][key]['visibility']}\n"
        'Actual RGB with projected mesh convex hulls; hulls do not represent pixelwise surface visibility',fontsize=10)
    fig.tight_layout();fig.savefig(out);plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('experiment','data-root','index-root','out'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--cases-per-category',type=int,default=4);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);rows,manifests=read_evaluation(a.experiment)
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    keys=[k for k in sorted(rows[ARMS[0]]) if not rows[ARMS[0]][k]['initialization']]
    table=[]
    for key in keys:
        ref=rows[ARMS[0]][key];s=streams[key[0]];d=s['mesh_diameter']
        entry=dict(stream_id=key[0],frame=key[1],object_id=ref['object_id'],visibility=ref['visibility'],diameter_m=d)
        for arm in ARMS:
            r=rows[arm][key]
            entry.update({f'{arm}_{name}':r[name] for name in ('center_mm','rotation_deg','add_01','adds_01','adds_005')})
            entry[f'{arm}_adds_d']=r['adds_m']/d;entry[f'{arm}_add_d']=r['add_m']/d
            entry[f'{arm}_support']=r.get('observation_support');entry[f'{arm}_anchors_read']=r.get('anchors_read')
        entry['joint_gain_adds_d']=entry['R0K0_adds_d']-entry['R1K1_adds_d'];table.append(entry)
    with (a.out/'paired_frames.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    occluded=[r for r in table if r['visibility'] is not None and r['visibility']<.5]
    # Per-stream selection avoids filling a page with neighboring frames from
    # one failure. All frame scores remain in paired_frames.csv.
    def select(records,score):
        chosen=[];seen=set()
        for r in sorted(records,key=score,reverse=True):
            if r['stream_id'] in seen:continue
            chosen.append(r);seen.add(r['stream_id'])
            if len(chosen)==a.cases_per_category:break
        return chosen
    selections={'persistent_occlusion_failure':select(occluded,lambda r:min(r[f'{arm}_adds_d'] for arm in ARMS)),
        'joint_improvement':select([r for r in occluded if r['joint_gain_adds_d']>0],lambda r:r['joint_gain_adds_d']),
        'joint_regression':select([r for r in occluded if r['joint_gain_adds_d']<0],lambda r:-r['joint_gain_adds_d'])}
    for category,records in selections.items():
        for i,r in enumerate(records):
            name=f'{category}_{i:02d}.png';key=(r['stream_id'],r['frame']);figure_case(key,rows,streams[key[0]],a.data_root,a.index_root,a.out/name);r['figure']=name
    per_object={}
    for obj in sorted(set(r['object_id'] for r in table)):
        per_object[obj]={}
        for population,rs in [('all',[r for r in table if r['object_id']==obj]),('visibility_lt_05',[r for r in occluded if r['object_id']==obj])]:
            per_object[obj][population]=dict(frames=len(rs),arms={arm:{metric:float(np.mean([r[f'{arm}_{metric}'] for r in rs])) if rs else None
                for metric in ('add_01','adds_01','adds_005','center_mm','rotation_deg')} for arm in ARMS})
    report=dict(completed=True,frames=len(table),occluded_frames=len(occluded),per_object=per_object,selections=selections,
        initialization=manifests[ARMS[0]]['initial_pose_source'],scope='Validation hardcase diagnosis; no test data, hand annotations or FP. Selection rules fixed in this script.',
        figures='Actual frame photos plus mathematical pose projections; green GT, red prediction; not manually adjudicated labels',
        visual_review_completed=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
