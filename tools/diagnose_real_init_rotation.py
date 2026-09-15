"""Read-only rotation-update audit of sealed real-initialized validation trajectories."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha
from lip.geometry.so3 import angle,log


def summarize(rows):
    objects=defaultdict(list)
    for r in rows:objects[r['object_id']].append(r)
    fields=('before_deg','after_deg','step_deg','error_reduction_deg','direction_cosine','step_over_needed','wrong_direction','improved')
    return dict(frames=len(rows),streams=len({r['stream_id'] for r in rows}),objects=len(objects),
        object_macro={field:float(np.mean([np.mean([r[field] for r in rs if r[field] is not None]) for rs in objects.values() if any(r[field] is not None for r in rs)])) if any(r[field] is not None for r in rows) else None for field in fields},
        frame_micro_medians={field:float(np.median([r[field] for r in rows if r[field] is not None])) if any(r[field] is not None for r in rows) else None for field in fields})


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',required=True,action='append');p.add_argument('--index-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);folders=dict(x.split('=',1) for x in a.evaluation)
    assert len(folders)==len(a.evaluation) and 'residual' in folders
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines()) if s['split']=='val'}
    data={};manifests={};initial={}
    for name,folder in folders.items():
        path=Path(folder);m=json.loads((path/'manifest.json').read_text());assert m['completed'] and m['population_verified'] and m['split']=='val' and m['frames']==23200
        assert sha(path/'predictions.jsonl')==m['predictions_sha256']
        rows=list(map(json.loads,(path/'predictions.jsonl').read_text().splitlines()));data[name]={(r['stream_id'],r['frame_index']):r for r in rows};manifests[name]=m
        assert len(data[name])==23200
    ref=data['residual'];ident=manifests['residual']
    for name,m in manifests.items():
        assert all(m[k]==ident[k] for k in ('initializers_sha256','split_hash','mesh_hash'))
        assert data[name].keys()==ref.keys()
        for key,r in ref.items():
            if r['initialization']:
                assert data[name][key]['pose_original']==r['pose_original'];initial[r['stream_id']]=r
    bins=((0,15),(15,45),(45,90),(90,181));records=[];gt_hashes={};max_reproduction=0.
    for sid,s in sorted(streams.items()):
        if sid not in initial:continue
        ir=initial[sid];first=ir['frame_index'];init_error=ir['rotation_deg']
        label=next(f'{lo}_{hi}' for lo,hi in bins if lo<=init_error<hi)
        gt_hashes[s['pose_cache']]=sha(a.index_root/s['pose_cache'])
        with np.load(a.index_root/s['pose_cache']) as z:
            assert np.array_equal(z['frames'],np.arange(s['num_frames']));gt=z['poses'][:,:3,:3].copy()
        for name,rs in data.items():
            for frame in range(first+1,min(first+9,s['num_frames'])):
                row=rs[(sid,frame)];prev=rs[(sid,frame-1)]
                assert row['pose_centered'] is not None and prev['pose_centered'] is not None
                old=np.asarray(prev['pose_centered'],dtype='f4')[:3,:3];new=np.asarray(row['pose_centered'],dtype='f4')[:3,:3]
                actual=float(angle(torch.from_numpy(new@gt[frame].T))*180/torch.pi)
                max_reproduction=max(max_reproduction,abs(actual-row['rotation_deg']));assert abs(actual-row['rotation_deg'])<1e-4
                # Diagnostic tangent vectors use double arithmetic; original
                # recorded FP32 rotation scores are independently reproduced above.
                old,new,target=[torch.from_numpy(x).double() for x in (old,new,gt[frame])]
                need=log(target@old.T);delta=log(new@old.T);n=float(need.norm());d=float(delta.norm());cos=float(torch.dot(need,delta)/(n*d)) if n>np.deg2rad(1.) and d>np.deg2rad(.01) else None
                before=float(angle(old@target.T)*180/torch.pi);after=row['rotation_deg']
                records.append(dict(method=name,stream_id=sid,object_id=s['object_id'],physical_sequence=row['physical_sequence'],frame_index=frame,update=frame-first,
                    initial_rotation_deg=init_error,initial_rotation_bin=label,initial_bad=ir['initial_bad'],visibility=row['visibility'],
                    before_deg=before,after_deg=after,step_deg=d*180/np.pi,error_reduction_deg=before-after,direction_cosine=cos,
                    step_over_needed=d/n if n>np.deg2rad(1.) else None,wrong_direction=bool(cos<0) if cos is not None else None,improved=after<before))
    report=dict(completed=True,scope='Posthoc diagnostic on sealed native s0 val, first eight updates. GT used only to analyze existing predictions; zero new inference, optimization or checkpoint selection. This is not a causal ablation.',
        checkpoints={n:m['checkpoint_sha256'] for n,m in manifests.items()},predictions_sha256={n:m['predictions_sha256'] for n,m in manifests.items()},
        initializers_sha256=ident['initializers_sha256'],gt_pose_cache_sha256=gt_hashes,rotation_score_max_abs_reproduction_error=max_reproduction,
        definitions=dict(before='Previous accepted rotation versus current GT; includes actual frame-to-frame object motion.',
            direction='Cosine between camera-frame logarithmic actual and required updates; only when required angle >1 degree and update >0.01 degree. Near-pi direction ambiguity remains.',
            step_ratio='Actual angular step / required angle; no implication that linear scaling would improve the pose.',
            populations='Bins fixed from common initial GT-relative angle; not each model current error. 319 legal initializers; a last-frame initializer has no next-frame update.'),
        initial_bins={},populations={})
    for lo,hi in bins:
        label=f'{lo}_{hi}';report['initial_bins'][label]=sum(lo<=r['rotation_deg']<hi for r in initial.values())
    populations={'all_first8':records,'initial_bad_first8':[r for r in records if r['initial_bad']],
        'all_first_update':[r for r in records if r['update']==1]}
    for label in report['initial_bins']:
        populations['angle_'+label+'_first8']=[r for r in records if r['initial_rotation_bin']==label]
        populations['angle_'+label+'_first_update']=[r for r in records if r['initial_rotation_bin']==label and r['update']==1]
    for pop,rs in populations.items():report['populations'][pop]={n:summarize([r for r in rs if r['method']==n]) for n in data}
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'frames.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report['frames_sha256']=sha(a.out/'frames.csv');report['entrypoint_sha256']=sha(Path(__file__))
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k in ('initial_bins','rotation_score_max_abs_reproduction_error','populations')},indent=2))


if __name__=='__main__':main()
