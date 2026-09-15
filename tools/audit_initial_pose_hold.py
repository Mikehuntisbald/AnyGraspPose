"""Posthoc initializer-hold and initial-motion audit on fixed occlusion frames."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def stationary_prefix(rotation_degrees,center_d,rotation_limit=2.,center_limit=.02):
    r=np.asarray(rotation_degrees);t=np.asarray(center_d)
    if r.shape!=t.shape or r.ndim!=1:raise ValueError('One rotation/center trace required')
    return np.logical_and.accumulate((r<=rotation_limit)&(t<=center_limit))


def summarize(rows,field):
    groups=defaultdict(list)
    for row in rows:groups[row['object_id']].append(row)
    return dict(frames=len(rows),successes=sum(r[field] for r in rows),
        object_macro=sum(sum(r[field] for r in group)/len(group) for group in groups.values())/len(groups) if groups else None)


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('runtime','origins','evaluation','index-root','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--actor-name',default='spatial_aug');a=p.parse_args();sys.path.insert(0,str(a.runtime/'src'))
    from lip.evaluation.metrics import errors
    from lip.geometry.so3 import center_pose,angle
    torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False)
    origins=json.loads(a.origins.read_text());manifest=json.loads((a.evaluation/'manifest.json').read_text())
    assert manifest['completed'] and manifest['checkpoint_sha256']==origins['checkpoints'][a.actor_name]
    raw=(a.evaluation/'predictions.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==origins['prediction_sha256'][a.actor_name]
    groups=defaultdict(list)
    for row in map(json.loads,raw.splitlines()):groups[row['stream_id']].append(row)
    events={(e['stream_id'],frame):e for e in origins['events'] for frame in e['frames']}
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    records=[];max_difference=0.
    for sid,rows in sorted(groups.items()):
        if not any((sid,r['frame_index']) in events for r in rows):continue
        rows=sorted(rows,key=lambda r:r['frame_index']);assert rows[0]['initialization'];s=streams[sid];assert s['split']=='val'
        with np.load(a.index_root/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
        with np.load(a.index_root/s['pose_cache']) as z:frames=z['frames'].copy();gt=center_pose(torch.from_numpy(z['poses'].copy()),torch.from_numpy(mesh['center']))
        assert list(map(int,frames))==[r['frame_index'] for r in rows]
        rotation=(angle(gt[:,:3,:3]@gt[0,:3,:3].T)*180/torch.pi).numpy()
        center=((gt[:,:3,3]-gt[0,:3,3]).norm(dim=-1)/float(mesh['diameter'])).numpy()
        primary=stationary_prefix(rotation,center);sensitivity=stationary_prefix(rotation,center,5.,.05)
        initializer=np.asarray(rows[0]['pose_centered'],dtype='f4')
        for i,row in enumerate(rows):
            key=(sid,row['frame_index'])
            if key not in events:continue
            event=events[key]
            actual=errors(row['pose_centered'],gt[i].numpy(),mesh['vertices'],float(mesh['diameter']),dtype='f4')
            difference=max(abs(actual[k]-row[k]) for k in ('add_m','adds_m','center_mm','rotation_deg'))
            max_difference=max(max_difference,difference);assert difference<1e-5,(key,difference)
            hold=errors(initializer,gt[i].numpy(),mesh['vertices'],float(mesh['diameter']),dtype='f4')
            records.append(dict(stream_id=sid,frame=row['frame_index'],object_id=row['object_id'],visibility=row['visibility'],
                category=event['category'],initial_image_clear=event['initial_image_clear'],initial_pose_adds005=event['initial_pose_adds005'],
                initial_stationary_prefix=bool(primary[i]),initial_stationary_prefix_loose=bool(sensitivity[i]),
                gt_rotation_from_initial_deg=float(rotation[i]),gt_center_from_initial_d=float(center[i]),
                lip_success=bool(row['adds_005']) and row['status']=='ok',hold_initial_success=bool(hold['adds_005']),
                lip_adds_d=actual['adds_m']/float(mesh['diameter']),hold_initial_adds_d=hold['adds_m']/float(mesh['diameter'])))
    pops={'all_occluded':records,'severe':[r for r in records if r['visibility']<.3],
        'no_prior_clear':[r for r in records if r['category']=='no_prior_three_clear_observations'],
        'no_prior_clear_severe':[r for r in records if r['category']=='no_prior_three_clear_observations' and r['visibility']<.3]}
    for name,rows in list(pops.items()):
        pops[name+'_before_motion']=[r for r in rows if r['initial_stationary_prefix']]
        pops[name+'_after_motion']=[r for r in rows if not r['initial_stationary_prefix']]
    summary={name:dict(lip=summarize(rows,'lip_success'),initializer_hold=summarize(rows,'hold_initial_success'),
        stationary_prefix_frames=sum(r['initial_stationary_prefix'] for r in rows),loose_stationary_prefix_frames=sum(r['initial_stationary_prefix_loose'] for r in rows)) for name,rows in pops.items()}
    (a.out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    report=dict(completed=True,scope='Posthoc fixed-occlusion-frame diagnostic on controlled s0 val. Holding the supplied noisy initializer never uses later GT; GT defines analysis bins only. This is not a deployable switch or a full benchmark score.',
        checkpoint_sha256=manifest['checkpoint_sha256'],prediction_sha256=hashlib.sha256(raw).hexdigest(),origins_sha256=hashlib.sha256(a.origins.read_bytes()).hexdigest(),
        metric='Same full-mesh-vertex FP32 ADD-S as archived evaluation; actor success also requires status ok. No sampled-metric substitution.',
        stationary_definition='Entire GT prefix remains within 2 degrees and .02d center displacement of initial GT pose. Once exceeded, later returns are still after-motion. Sensitivity: 5 degrees/.05d.',
        archived_metric_max_abs_difference=max_difference,populations=summary)
    (a.out/'audit.json').write_text(json.dumps(report,indent=2));lines=['# Initializer hold and occluded motion','',report['scope'],'',report['stationary_definition'],'',
        '| Population | Frames | LIP ADD-S@.05d (%) | Hold initializer (%) |','|---|---:|---:|---:|']
    for name,r in summary.items():
        if r['lip']['frames']:lines.append(f"| {name} | {r['lip']['frames']} | {100*r['lip']['object_macro']:.3f} | {100*r['initializer_hold']['object_macro']:.3f} |")
    (a.out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
