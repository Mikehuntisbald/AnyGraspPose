"""Paired physical-sequence bootstrap for a matched R x K factorial experiment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS=('R0K0','R1K0','R0K1','R1K1')
CONTRASTS={'R_at_K0':[-1,1,0,0],'R_at_K1':[0,0,-1,1],
    'K_at_R0':[-1,0,1,0],'K_at_R1':[0,-1,0,1],
    'R_average':[-.5,.5,-.5,.5],'K_average':[-.5,-.5,.5,.5],
    'interaction':[1,-1,-1,1]}
METRICS=('add_01','adds_01','adds_005','center_mm','rotation_deg','lost')


def episode_populations(rows):
    """GT visibility defines scoring populations only, never predictor inputs."""
    long_frames=set();episodes=[];streams={}
    for row in rows:streams.setdefault(row['stream_id'],[]).append(row)
    for sid,rs in streams.items():
        rs.sort(key=lambda r:r['frame_index']);i=0
        while i<len(rs):
            low=lambda r:r['visibility'] is not None and r['visibility']<.5
            if not low(rs[i]):i+=1;continue
            j=i+1
            while j<len(rs) and low(rs[j]):j+=1
            if j-i>8:
                long_frames.update((sid,r['frame_index']) for r in rs[i:j])
                recovery=[]
                for r in rs[j:j+10]:
                    if r['visibility'] is None or low(r):break
                    recovery.append((sid,r['frame_index']))
                episodes.append(dict(stream_id=sid,start=rs[i]['frame_index'],end=rs[j-1]['frame_index'],
                    length=j-i,post_keys=recovery,right_censored=len(recovery)<10))
            i=j
    return long_frames,episodes


def paired_summary(values,objects,sequences,weights):
    """values: frame x arm; macro-object estimator for each cluster resample."""
    obj=np.unique(objects);seq=np.unique(sequences)
    sums=np.zeros((len(seq),len(obj),values.shape[1]));counts=np.zeros((len(seq),len(obj)))
    si=np.searchsorted(seq,sequences);oi=np.searchsorted(obj,objects)
    np.add.at(sums,(si,oi),values);np.add.at(counts,(si,oi),1)
    def aggregate(w):
        numerator=np.einsum('ds,soa->doa',w,sums);denominator=w@counts
        ratio=np.divide(numerator,denominator[...,None],out=np.full_like(numerator,np.nan),where=denominator[...,None]>0)
        with np.errstate(invalid='ignore'):return np.nanmean(ratio,axis=1)
    point=aggregate(np.ones((1,len(seq))))[0]
    draws=aggregate(weights)
    return point,draws


def compare(root,draws=2000,seed=20260913):
    rows={};manifests={};inputs={}
    for arm in ARMS:
        folder=root/arm/'s0_val';manifests[arm]=json.loads((folder/'manifest.json').read_text())
        m=manifests[arm]
        assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
        assert m['fp_calls']==m['critic_calls']==0 and m['split']=='val'
        assert m['checkpoint_stage_step']==1000
        data=(folder/'predictions.jsonl').read_bytes();inputs[arm]=hashlib.sha256(data).hexdigest()
        rs=[r for r in map(json.loads,data.splitlines()) if not r['initialization']]
        rows[arm]={(r['stream_id'],r['frame_index']):r for r in rs}
        assert len(rows[arm])==len(rs)
    base=manifests[ARMS[0]]
    for m in manifests.values():
        assert all(m.get(k)==base.get(k) for k in ('split_hash','mesh_hash','source_sha256','initial_pose_source','initial_poses_sha256','initialization_uses_gt_pose'))
        assert m.get('initial_poses_sha256'), 'This matched experiment requires an explicit fixed initializer artifact'
    keys=sorted(rows[ARMS[0]])
    for arm in ARMS:
        assert set(keys)==set(rows[arm])
        for k in keys:
            assert rows[arm][k]['visibility']==rows[ARMS[0]][k]['visibility']
    reference=[rows[ARMS[0]][k] for k in keys];long_keys,episodes=episode_populations(reference)
    sequences=np.array(['/'.join(k[0].split('/')[:2]) for k in keys]);objects=np.array([r['object_id'] for r in reference])
    clusters=np.unique(sequences);rng=np.random.default_rng(seed)
    # Shared cluster draws across all arms, metrics and visibility populations.
    weights=rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=draws)
    populations={'all':np.ones(len(keys),dtype=bool),
        'visibility_lt_05':np.array([r['visibility'] is not None and r['visibility']<.5 for r in reference]),
        'visibility_lt_03':np.array([r['visibility'] is not None and r['visibility']<.3 for r in reference]),
        'visibility_ge_05':np.array([r['visibility'] is not None and r['visibility']>=.5 for r in reference]),
        'long_occlusion_gt_8_frames':np.array([k in long_keys for k in keys])}
    report=dict(completed=True,frames_including_initialization=23200,tracked_frames=len(keys),streams=320,
        initial_pose_source=base['initial_pose_source'],initialization_uses_gt_pose=base['initialization_uses_gt_pose'],
        initial_poses_sha256=base['initial_poses_sha256'],prediction_sha256=inputs,
        checkpoints={arm:manifests[arm]['checkpoint_sha256'] for arm in ARMS},
        scope='Controlled s0 validation, zero FP, one training seed; not BOP AR or a non-GT initialization claim',
        visibility_definition='GT segmentation overlap with projected GT silhouette, inherited validation proxy; not official BOP visib_fract',
        estimator='Object macro; paired bootstrap resamples physical sequences including their cameras together',
        bootstrap=dict(draws=draws,seed=seed,physical_sequences=len(clusters),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest()),populations={},recovery={},resources={})
    for arm in ARMS:
        logs=[json.loads(line) for line in (root/arm/'train/rank0.jsonl').read_text().splitlines()]
        measured=[rows[arm][k]['inference_seconds'] for k in keys]
        report['resources'][arm]=dict(train_steps=len(logs),median_train_step_seconds=float(np.median([r['seconds'] for r in logs])),
            peak_allocated_rank0_bytes=max(r['peak_allocated'] for r in logs),
            observed_eval_median_seconds=float(np.median(measured)),eval_timing_includes_frame_io=True,
            timing_caveat='Concurrent factorial runs; not a standalone latency benchmark',
            final_reliability_strength=logs[-1]['reliability_strength'],final_update_strength=logs[-1]['update_strength'],
            mean_anchors_read=float(np.mean([rows[arm][k].get('anchors_read',0) for k in keys])))
    csv_rows=[]
    for name,mask in populations.items():
        target=dict(frames=int(mask.sum()),metrics={});report['populations'][name]=target
        if not mask.any():continue
        present=np.isin(clusters,np.unique(sequences[mask]));w=weights[:,present]
        for metric in METRICS:
            scale=100 if metric in ('add_01','adds_01','adds_005','lost') else 1
            values=np.array([[rows[arm][k][metric]*scale for arm in ARMS] for k in keys])[mask]
            point,boot=paired_summary(values,objects[mask],sequences[mask],w)
            result=dict(arms=dict(zip(ARMS,point.tolist())),contrasts={})
            for contrast,coeff in CONTRASTS.items():
                delta=boot@coeff;finite=delta[np.isfinite(delta)]
                result['contrasts'][contrast]=dict(delta=float(point@coeff),ci95=np.quantile(finite,[.025,.975]).tolist(),valid_draws=len(finite))
            target['metrics'][metric]=result
            for sid in np.unique(sequences[mask]):
                seq_select=sequences[mask]==sid
                for obj in np.unique(objects[mask][seq_select]):
                    sel=seq_select&(objects[mask]==obj)
                    for ai,arm in enumerate(ARMS):csv_rows.append(dict(population=name,physical_sequence=sid,object_id=int(obj),arm=arm,metric=metric,
                        frame_count=int(sel.sum()),sum=float(values[sel,ai].sum()),mean=float(values[sel,ai].mean())))
    # Recovery is conditioned on failure at the final occluded frame. Report
    # already-successful episodes separately; censored windows are never failures.
    for arm in ARMS:
        complete=[e for e in episodes if not e['right_censored']]
        failed=[e for e in complete if not rows[arm][(e['stream_id'],e['end'])]['adds_01']]
        times=[]
        for e in failed:
            success=next((i+1 for i,k in enumerate(e['post_keys']) if rows[arm][k]['adds_01'] and rows[arm][k]['status']=='ok'),None)
            if success is not None:times.append(success)
        report['recovery'][arm]=dict(long_episodes=len(episodes),complete_10_frame_windows=len(complete),right_censored=len(episodes)-len(complete),
            failed_at_occlusion_end=len(failed),already_successful_at_end=len(complete)-len(failed),recovered_within_10=len(times),
            recovery_rate=len(times)/len(failed) if failed else None,median_successful_recovery_frames=float(np.median(times)) if times else None)
    (root/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (root/'paired_physical_sequences.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    lines=['# Observation reliability x keyframe memory','',''+report['scope'],
        '',f"Initialization: {report['initial_pose_source']}. Initial frames excluded. Intervals describe validation-population variability, not training-seed variability.",
        '', '| Population / metric | R0K0 | R1K0 | R0K1 | R1K1 | Interaction (95% CI) |','|---|---:|---:|---:|---:|---:|']
    for pop in populations:
        for metric in ('add_01','adds_005'):
            r=report['populations'][pop]['metrics'].get(metric)
            if not r:continue
            d=r['contrasts']['interaction'];low,high=d['ci95']
            lines.append(f"| {pop} / {metric} (%) | "+' | '.join(f'{v:.3f}' for v in r['arms'].values())+f" | {d['delta']:+.3f} [{low:+.3f}, {high:+.3f}] |")
    lines+=['','Interaction = R1K1 - R1K0 - R0K1 + R0K0. Positive is better for success rates. Full conditional effects, errors and censored recovery counts are in comparison.json.',
        '',report['visibility_definition']]
    (root/'report.md').write_text('\n'.join(lines)+'\n');return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--draws',type=int,default=2000);a=p.parse_args()
    compare(a.out,a.draws)


if __name__=='__main__':main()
