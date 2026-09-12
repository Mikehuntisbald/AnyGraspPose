"""Score saved pre/post-FP poses on identical hybrid histories; CPU only, no reruns."""
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
import torch
from lip.geometry.so3 import center_pose
from lip.evaluation.metrics import errors,summarize

R=Path('/mnt/why/dexycb_lip');INDEX=R/'cache/dexycb_s0'
OUT=R/'runs/fp_transition_analysis_21530_31000'
KEYS=['center_mm','rotation_deg','add_m','adds_m','add_005','add_01','adds_005','adds_01']

def initialize():
    global STREAMS
    torch.set_num_threads(1)
    STREAMS={s['stream_id']:s for s in map(json.loads,(INDEX/'streams.jsonl').read_text().splitlines()) if s['split']=='val'}

def analyze(job):
    step,sid,rows=job;s=STREAMS[sid]
    with np.load(INDEX/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
    with np.load(INDEX/s['pose_cache']) as z:frames=z['frames'].copy();original=z['poses'].copy()
    gt=center_pose(torch.from_numpy(original),torch.from_numpy(mesh['center'])).numpy()
    lookup={int(f):p for f,p in zip(frames,gt)};d=float(mesh['diameter']);result=[];max_check=0.
    for i,r in enumerate(rows):
        target=lookup[r['frame_index']];after={k:r[k] for k in KEYS}
        update='lip_proposal_centered' in r
        if update:
            pose=np.asarray(r['lip_proposal_centered']);before=errors(pose,target,mesh['vertices'],d)
        else:
            assert r['frame_index']==int(frames[0]);before=after.copy();pose=np.asarray(r['pose_centered'])
        after_pose=np.asarray(r['pose_centered'])
        if i in (0,1,len(rows)//2,len(rows)-1):
            check=errors(after_pose,target,mesh['vertices'],d)
            max_check=max(max_check,max(abs(check[k]-after[k]) for k in KEYS))
            assert all(abs(check[k]-after[k])<1e-6 for k in KEYS),'Recomputed post-FP metric mismatch'
        result.append(dict(step=step,stream_id=sid,object_id=r['object_id'],frame_index=r['frame_index'],
            visibility_bin=r['visibility_bin'],visibility=r['visibility'],moving=r['moving'],fp_update=update,
            diameter_m=d,before=before,after=after,
            shift_xyz_mm=((after_pose[:3,3]-pose[:3,3])*1000).tolist(),
            bias_before_xyz_mm=((pose[:3,3]-target[:3,3])*1000).tolist(),
            bias_after_xyz_mm=((after_pose[:3,3]-target[:3,3])*1000).tolist()))
    return result,max_check

def statistics(rows):
    n=len(rows);assert n
    before=np.array([r['before']['adds_m']/r['diameter_m'] for r in rows]);after=np.array([r['after']['adds_m']/r['diameter_m'] for r in rows])
    transitions={}
    for metric in ('add_01','adds_01'):
        a=np.array([bool(r['before'][metric]) for r in rows]);b=np.array([bool(r['after'][metric]) for r in rows])
        transitions[metric]=dict(before_pass=int(a.sum()),after_pass=int(b.sum()),pass_to_fail=int((a&~b).sum()),fail_to_pass=int((~a&b).sum()),both_fail=int((~a&~b).sum()),both_pass=int((a&b).sum()))
    return dict(frames=n,transitions=transitions,
        meaningful_improve=int((after<before-.005).sum()),meaningful_harm=int((after>before+.005).sum()),
        within_margin=int((np.abs(after-before)<=.005).sum()),
        mean_error_before_d=float(before.mean()),mean_error_after_d=float(after.mean()),
        before={k:float(np.mean([r['before'][k] for r in rows])) for k in KEYS},
        after={k:float(np.mean([r['after'][k] for r in rows])) for k in KEYS},
        mean_shift_xyz_mm=np.mean([r['shift_xyz_mm'] for r in rows],axis=0).tolist(),
        median_shift_xyz_mm=np.median([r['shift_xyz_mm'] for r in rows],axis=0).tolist(),
        mean_bias_before_xyz_mm=np.mean([r['bias_before_xyz_mm'] for r in rows],axis=0).tolist(),
        mean_bias_after_xyz_mm=np.mean([r['bias_after_xyz_mm'] for r in rows],axis=0).tolist())

def main():
    OUT.mkdir(exist_ok=False)
    allrows=[];checks=[];started=time.time()
    with concurrent.futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for step in (21530,31000):
            p=R/f'runs/lip_fp_{step}/lip_fp/predictions.jsonl';groups={}
            for r in map(json.loads,p.read_text().splitlines()):groups.setdefault(r['stream_id'],[]).append(r)
            jobs=[(step,sid,rows) for sid,rows in groups.items()]
            futures=[pool.submit(analyze,j) for j in jobs];rows=[]
            for f in concurrent.futures.as_completed(futures):
                batch,check=f.result();rows.extend(batch);checks.append(check)
                (OUT/'status.json').write_text(json.dumps(dict(phase='scoring',step=step,frames=len(rows),expected=23200,seconds=time.time()-started)))
            assert len(rows)==23200 and sum(r['fp_update'] for r in rows)==22880
            rows.sort(key=lambda r:(r['stream_id'],r['frame_index']));allrows.extend(rows)
            (OUT/f'frames_{step}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            conditioned={side:summarize([dict(r[side],**{k:r[k] for k in ('object_id','stream_id','visibility_bin','moving')}) for r in rows]) for side in ('before','after')}
            plain=json.loads((R/f'runs/lip_fp_{step}/lip/metrics.json').read_text())
            hybrid=json.loads((R/f'runs/lip_fp_{step}/lip_fp/metrics.json').read_text())
            for k in KEYS:assert abs(conditioned['after']['macro_object'][k]-hybrid['macro_object'][k])<1e-10
            updates=[r for r in rows if r['fp_update']]
            grouped={}
            for key in ('object_id','visibility_bin','moving'):
                groups={}
                for r in updates:groups.setdefault(str(r[key]),[]).append(r)
                grouped[key]={k:statistics(v) for k,v in groups.items()}
            bins={}
            for lo,hi,label in [(0,.02,'0-.02d'),(.02,.05,'.02-.05d'),(.05,.1,'.05-.1d'),(.1,float('inf'),'>=.1d')]:
                rr=[r for r in updates if lo<=r['before']['adds_m']/r['diameter_m']<hi]
                if rr:bins[label]=statistics(rr)
            A=plain['macro_object'];B=conditioned['before']['macro_object'];C=conditioned['after']['macro_object']
            report=dict(step=step,all_frames=23200,actual_FP_updates=22880,margin_d=.005,
                definitions=dict(A='Pure LIP closed loop on its own history',B='Pre-FP LIP proposals on hybrid history; conditional diagnostic, not a deployed tracker',C='Post-FP output and accepted hybrid history'),
                decomposition={k:dict(A=A[k],B=B[k],C=C[k],history_associated=B[k]-A[k],immediate_FP=C[k]-B[k],total=C[k]-A[k]) for k in KEYS},
                calls=statistics(updates),by_before_adds_error=bins,groups=grouped,
                before_on_hybrid_history=conditioned['before'],after_on_hybrid_history=conditioned['after'],
                input_predictions_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                scope='Full saved validation trajectories, all mesh vertices; exact per-frame before/after FP comparison. History association is an arithmetic decomposition, not an independent history intervention. No model update or new FP inference.')
            (OUT/f'report_{step}.json').write_text(json.dumps(report,indent=2))
    (OUT/'status.json').write_text(json.dumps(dict(phase='complete',frames=len(allrows),max_recomputed_after_metric_difference=max(checks),seconds=time.time()-started),indent=2))

if __name__=='__main__':main()
