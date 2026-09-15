"""Same-checkpoint branch intervention and post-inference local rotation accounting."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lip.engine.stream_checkpoint import sha
from lip.geometry.so3 import angle, log
from compare_rk_ablation import paired_summary


def summarize_pair(values, objects, sequences, clusters, weights, names):
    local = weights[:, np.isin(clusters, np.unique(sequences))]
    point, draws = paired_summary(values, objects, sequences, local)
    delta = draws[:, 0] - draws[:, 1]; delta = delta[np.isfinite(delta)]
    return dict(frames=len(values), values=dict(zip(names, point.tolist())),
                delta=float(point[0]-point[1]), ci95=np.quantile(delta, [.025,.975]).tolist(), valid_draws=len(delta))


def local_rotation(parent, after, target, correction):
    parent, after, target, correction = [torch.as_tensor(v, dtype=torch.float64) for v in (parent, after, target, correction)]
    before = angle(parent @ target.transpose(-1,-2)) * 180/torch.pi
    final = angle(after @ target.transpose(-1,-2)) * 180/torch.pi
    need = log(target @ parent.transpose(-1,-2)); magnitude = correction.norm(dim=-1)
    cosine = (need*correction).sum(-1)/(need.norm(dim=-1)*magnitude).clamp_min(1e-30)
    valid = (need.norm(dim=-1)>np.deg2rad(1.)) & (magnitude>np.deg2rad(.01))
    return dict(before_deg=before.numpy(), after_deg=final.numpy(), reduction_deg=(before-final).numpy(),
                step_deg=(magnitude*180/torch.pi).numpy(), cosine=cosine.numpy(), direction_valid=valid.numpy())


def write_csv(path, rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    root=Path(__file__).resolve().parents[1]; r=root/'runs/ablation'; out=r/'analysis'; out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    assert json.loads((r/'status.json').read_text())['phase']=='completed'
    folders={n:r/n/'scored' for n in ('learned','zero')}
    folders['control']=Path('/mnt/why/dexycb_lip/intraframe_alignment_20260915/runs/alignment_val/control/scored')
    data={}; manifests={}
    for n, folder in folders.items():
        m=json.loads((folder/'manifest.json').read_text()); manifests[n]=m
        assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
        assert m['fp_calls']==0 and not m['initialization_uses_gt_pose'] and sha(folder/'predictions.jsonl')==m['predictions_sha256']
        data[n]={(x['stream_id'],x['frame_index']):x for x in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines())}
    assert manifests['learned']['checkpoint_sha256']==manifests['zero']['checkpoint_sha256']
    assert manifests['learned']['model_tensors_sha256']==manifests['zero']['model_tensors_sha256']
    keys=sorted(data['learned']); rows=[data['learned'][k] for k in keys]
    for n in data:
        assert data[n].keys()==data['learned'].keys()
        for key,row in zip(keys,rows):
            other=data[n][key]
            assert all(other[k]==row[k] for k in ('initialization','initial_bad','updates_after_initialization','visibility','object_id','physical_sequence'))
            if row['initialization']: assert other['pose_original']==row['pose_original']
    objects=np.array([v['object_id'] for v in rows]); sequences=np.array([v['physical_sequence'] for v in rows]); clusters=np.unique(sequences)
    weights=np.random.default_rng(20260915).multinomial(len(clusters),np.ones(len(clusters))/len(clusters),size=2000)
    updates=np.array([v['updates_after_initialization'] is not None and v['updates_after_initialization']>0 for v in rows])
    first8=np.array([v['updates_after_initialization'] is not None and 0<v['updates_after_initialization']<=8 for v in rows])
    bad=np.array([v['initial_bad'] is True for v in rows])
    masks=dict(all=np.ones(len(rows),bool), updates=updates, bad_initial_first8=first8&bad,
        visibility_lt_03=np.array([v['visibility'] is not None and v['visibility']<.3 for v in rows]),
        initial_good=updates&~bad, initial_bad=updates&bad)
    comparisons={}; flat=[]; paired=[]
    for names in [('learned','zero'),('zero','control')]:
        pair='_vs_'.join(names); comparisons[pair]={}
        for pop,mask in masks.items():
            comparisons[pair][pop]={}
            for metric in ('add_01','adds_005','rotation_deg','center_mm'):
                use=mask&np.array([all(data[n][k][metric] is not None for n in names) for k in keys])
                scale=100 if metric in ('add_01','adds_005') else 1
                selected=[k for k,yes in zip(keys,use) if yes]
                values=np.array([[data[n][k][metric]*scale for n in names] for k in selected])
                result=summarize_pair(values,objects[use],sequences[use],clusters,weights,names)
                comparisons[pair][pop][metric]=result
                flat.append(dict(pair=pair,population=pop,metric=metric,**{n:result['values'][n] for n in names},delta=result['delta'],lower=result['ci95'][0],upper=result['ci95'][1]))
    # Uniform columns across both comparisons.
    for row in flat:
        for n in data: row.setdefault(n,None)
    for key,row in zip(keys,rows):
        item={k:row[k] for k in ('stream_id','frame_index','object_id','physical_sequence','visibility','initial_bad','updates_after_initialization')}
        for n in data:
            for metric in ('add_01','adds_005','rotation_deg','center_mm'): item[n+'_'+metric]=data[n][key][metric]
        paired.append(item)
    index=Path('/mnt/why/dexycb_lip/cache/dexycb_s0'); streams={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines()) if s['split']=='val'}
    gt={}; gt_hashes={}
    for sid,s in streams.items():
        path=index/s['pose_cache']; gt_hashes[s['pose_cache']]=sha(path)
        with np.load(path) as z: gt[sid]=z['poses'][:,:3,:3].copy()
    indices=np.flatnonzero(updates); active=[rows[i] for i in indices]
    parent=np.array([v['pose_before_alignment'] for v in active]); after=np.array([v['pose_centered'] for v in active])
    assert np.array_equal(parent[:,:3,3],after[:,:3,3])
    result=local_rotation(parent[:,:3,:3],after[:,:3,:3],np.array([gt[v['stream_id']][v['frame_index']] for v in active]),np.array([v['unapplied_rotation_vec'] for v in active]))
    max_error=float(np.max(np.abs(result['after_deg']-np.array([v['rotation_deg'] for v in active]))))
    assert max_error<1e-3, max_error
    local=[]; local_summary={}
    for j,row in enumerate(active):
        item={k:row[k] for k in ('stream_id','frame_index','object_id','physical_sequence','initial_bad','updates_after_initialization')}
        item.update({k:float(v[j]) for k,v in result.items() if k not in ('direction_valid','cosine')})
        item['direction_cosine']=float(result['cosine'][j]) if result['direction_valid'][j] else None
        local.append(item)
    for pop,mask in masks.items():
        use=mask[indices]
        stats=summarize_pair(np.stack([result['after_deg'][use],result['before_deg'][use]],1),objects[indices][use],sequences[indices][use],clusters,weights,('after','before'))
        vs=[v for v,take in zip(local,use) if take]
        macro=lambda field:float(np.mean([np.mean([v[field] for v in vs if v['object_id']==o]) for o in sorted({v['object_id'] for v in vs})]))
        stats.update(mean_step_deg=macro('step_deg'),micro_mean_step_deg=float(result['step_deg'][use].mean()),
                     micro_fraction_improved=float((result['reduction_deg'][use]>1e-5).mean()),micro_fraction_worsened=float((result['reduction_deg'][use]<-1e-5).mean()))
        local_summary[pop]=stats
    write_csv(out/'comparisons.csv',flat); write_csv(out/'paired_frames.csv',paired); write_csv(out/'local_rotation_frames.csv',local)
    report=dict(completed=True,checkpoints={n:m['checkpoint_sha256'] for n,m in manifests.items()},
        predictions_sha256={n:m['predictions_sha256'] for n,m in manifests.items()},same_tensor_hash=manifests['learned']['model_tensors_sha256'],
        learned_full_trajectory_exact_reproduction=True,frames=23200,streams=320,updates=int(updates.sum()),
        missing_frames=sum(v['pose_centered'] is None for v in rows),comparisons=comparisons,local=local_summary,
        local_rotation_score_max_abs_reproduction_error=max_error,branch_center_bitwise_unchanged=True,
        bootstrap=dict(draws=2000,seed=20260915,physical_sequences=len(clusters),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),multiplicity_adjusted=False),
        gt_pose_hashes=gt_hashes,entrypoint_sha256=sha(Path(__file__)),
        scope='Same-checkpoint learned-vs-zero is a causal branch intervention including recursive feedback. Zero-vs-control compares different trained weights and does not identify compensation. Local after-vs-before holds the recorded learned-trajectory current inputs fixed. GT read only after inference. No seed stability, official BOP AR, or SOTA claim.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(comparisons=comparisons,local=local_summary),indent=2))


if __name__=='__main__':main()
