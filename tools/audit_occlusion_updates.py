"""Read-only, one-step update audit on frozen validation trajectories.

Hold/motion candidates share the saved LIP history. They are counterfactual
one-step diagnostics, not recursively evaluated trackers or benchmark scores.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sampled_add(pose,gt,points,diameter):
    a=points@pose[:3,:3].T+pose[:3,3];b=points@gt[:3,:3].T+gt[:3,3]
    return float(np.linalg.norm(a-b,axis=-1).mean()/diameter)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);p.add_argument('--index-root',required=True,type=Path)
    p.add_argument('--arm',default='R1K1');p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    folder=a.experiment/a.arm/'s0_val';m=json.loads((folder/'manifest.json').read_text());assert m['completed'] and m['population_verified']
    groups={}
    for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()):groups.setdefault(r['stream_id'],[]).append(r)
    rows=[]
    for sid,rs in groups.items():
        rs.sort(key=lambda r:r['frame_index']);s=streams[sid]
        with np.load(a.index_root/s['mesh_cache']) as z:points=z['points'].astype('f8');center=z['center'].astype('f8');diameter=float(z['diameter'])
        with np.load(a.index_root/s['pose_cache']) as z:gtposes={int(f):t.astype('f8') for f,t in zip(z['frames'],z['poses'])}
        for i,r in enumerate(rs):
            if r['initialization']:continue
            if i==0:raise ValueError('Missing initialization record')
            gt=gtposes[r['frame_index']].copy();gt[:3,3]+=gt[:3,:3]@center
            base=np.array(rs[i-1]['pose_centered']);observed=np.array(r['pose_centered']);motion=base.copy()
            if i>1:
                prev=np.array(rs[i-2]['pose_centered']);dt=rs[i-1]['timestamp']-rs[i-2]['timestamp']
                ratio=(r['timestamp']-rs[i-1]['timestamp'])/dt
                motion[:3,3]+=ratio*(base[:3,3]-prev[:3,3])
                increment=Rotation.from_matrix(base[:3,:3]@prev[:3,:3].T).as_rotvec()*ratio
                motion[:3,:3]=Rotation.from_rotvec(increment).as_matrix()@base[:3,:3]
            errors={name:sampled_add(t,gt,points,diameter) for name,t in [('hold',base),('lip',observed),('motion',motion)]}
            rows.append(dict(stream_id=sid,frame=r['frame_index'],object_id=r['object_id'],visibility=r['visibility'],support=r.get('observation_support'),
                **{name+'_add_d':v for name,v in errors.items()},hold_center_mm=float(np.linalg.norm(base[:3,3]-gt[:3,3])*1000),
                lip_center_mm=r['center_mm'],motion_center_mm=float(np.linalg.norm(motion[:3,3]-gt[:3,3])*1000),
                harmful_update=errors['lip']>errors['hold']+.005,helpful_update=errors['lip']<errors['hold']-.005,
                motion_better=errors['motion']<errors['lip']-.005,hold_better=errors['hold']<errors['lip']-.005))
    with (a.out/'one_step_candidates.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pops={'all':rows,'visibility_lt_05':[r for r in rows if r['visibility'] is not None and r['visibility']<.5],
        'visibility_lt_03':[r for r in rows if r['visibility'] is not None and r['visibility']<.3],
        'visibility_ge_05':[r for r in rows if r['visibility'] is not None and r['visibility']>=.5]}
    report=dict(completed=True,arm=a.arm,checkpoint_sha256=m['checkpoint_sha256'],initialization=m['initial_pose_source'],
        scope='One-step candidates from the same saved LIP history; sampled ADD on 512 surface points; micro summaries; not recursive tracking accuracy or an oracle deployment claim',populations={})
    for name,rs in pops.items():
        if not rs:continue
        q=[r for r in rs if r['support'] is not None]
        report['populations'][name]=dict(frames=len(rs),harmful_update_fraction=float(np.mean([r['harmful_update'] for r in rs])),
            helpful_update_fraction=float(np.mean([r['helpful_update'] for r in rs])),motion_better_fraction=float(np.mean([r['motion_better'] for r in rs])),
            mean_add_change_d=float(np.mean([r['lip_add_d']-r['hold_add_d'] for r in rs])),
            mean_center_change_mm=float(np.mean([r['lip_center_mm']-r['hold_center_mm'] for r in rs])),
            support_mean=float(np.mean([r['support'] for r in q])) if q else None)
    q=[r for r in rows if r['support'] is not None and r['visibility'] is not None]
    report['support_visibility_auc']=float(roc_auc_score([r['visibility']>=.5 for r in q],[r['support'] for r in q]))
    report['support_safe_update_auc']=float(roc_auc_score([not r['harmful_update'] for r in q],[r['support'] for r in q]))
    diagnostic=json.loads((a.experiment/'hardcases/diagnostic.json').read_text());seen=set()
    for category,selected in diagnostic['selections'].items():
        for item in selected[:2]:
            sid=item['stream_id']
            if sid in seen:continue
            seen.add(sid);rs=[r for r in rows if r['stream_id']==sid];x=[r['frame'] for r in rs]
            fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
            for candidate in ['hold','lip','motion']:axes[0].plot(x,[r[candidate+'_add_d'] for r in rs],label=candidate)
            axes[0].set_ylabel('Sampled ADD / d');axes[0].legend()
            axes[1].plot(x,[r['visibility'] if r['visibility'] is not None else np.nan for r in rs],label='GT visibility');axes[1].plot(x,[r['support'] if r['support'] is not None else np.nan for r in rs],label='Depth support');axes[1].legend();axes[1].set_ylim(-.05,1.05)
            axes[2].plot(x,[r['lip_add_d']-r['hold_add_d'] for r in rs]);axes[2].axhline(0,color='black',lw=.6);axes[2].set_ylabel('LIP minus hold ADD/d');axes[2].set_xlabel('Frame')
            fig.suptitle(sid+'\nOne-step counterfactuals from shared saved history, not full tracker ablations',fontsize=10)
            fig.tight_layout();fig.savefig(a.out/f'trajectory_{len(seen):02d}.png',dpi=140);plt.close(fig)
    (a.out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
