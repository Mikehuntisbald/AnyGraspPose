"""Paired candidate screening on the same full real-initialized s0 val streams."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_rk_ablation import paired_summary


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',action='append',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();folders=dict(x.split('=',1) for x in a.evaluation)
    assert {'control','smooth_rotation','residual'}<=set(folders);data={};manifests={}
    for name,value in folders.items():
        folder=Path(value);m=json.loads((folder/'manifest.json').read_text())
        assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
        assert m['split']=='val' and not m['initialization_uses_gt_pose'] and m['fp_calls']==0
        assert hashlib.sha256((folder/'predictions.jsonl').read_bytes()).hexdigest()==m['predictions_sha256']
        rows=list(map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()));data[name]={(r['stream_id'],r['frame_index']):r for r in rows}
        assert len(rows)==len(data[name])==23200;manifests[name]=m
    reference=data['control'];keys=sorted(reference);names=list(data)
    for name,rows in data.items():
        assert set(rows)==set(keys)
        assert all(manifests[name][k]==manifests['control'][k] for k in ('source_sha256','split_hash','mesh_hash','initializers_sha256'))
        for key,row in reference.items():
            other=rows[key]
            assert all(other[field]==row[field] for field in ('initialization','visibility','initial_bad','updates_after_initialization'))
            if row['initialization']:assert other['pose_centered']==row['pose_centered']
    rows=[reference[k] for k in keys];objects=np.asarray([r['object_id'] for r in rows]);sequences=np.asarray([r['physical_sequence'] for r in rows]);clusters=np.unique(sequences)
    weights=np.random.default_rng(20260915).multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=2000)
    masks=dict(all=np.ones(len(keys),bool),visibility_lt_05=np.asarray([r['visibility'] is not None and r['visibility']<.5 for r in rows]),
        visibility_lt_03=np.asarray([r['visibility'] is not None and r['visibility']<.3 for r in rows]),
        bad_initial_first8=np.asarray([bool(r['initial_bad'] and r['updates_after_initialization'] is not None and 0<r['updates_after_initialization']<=8) for r in rows]),
        initial_bad=np.asarray([bool(r['initial_bad'] and r['updates_after_initialization'] is not None and r['updates_after_initialization']>0) for r in rows]))
    comparisons=[('smooth_rotation','control'),('smooth_rotation','residual'),('control','residual')]
    result=dict(completed=True,frames=23200,streams=320,checkpoints={n:m['checkpoint_sha256'] for n,m in manifests.items()},
        initializers_sha256=manifests['control']['initializers_sha256'],populations={},
        scope='Full native s0 val, real PoseCNN first legal initialization, causal own-state tracking, zero FP. Object macro ADD thresholds and paired physical-sequence intervals. Missing initialization stays in the primary denominator. This is not test BOP AR.',
        bootstrap=dict(draws=2000,seed=20260915,physical_sequences=len(clusters),multiplicity_adjusted=False))
    exports=[]
    for pop,mask in masks.items():
        metrics={};result['populations'][pop]=dict(frames=int(mask.sum()),metrics=metrics)
        if not mask.any():continue
        for metric in ('add_01','adds_005','center_mm','rotation_deg'):
            use=mask.copy()
            if metric in ('center_mm','rotation_deg'):
                use &= np.asarray([all(data[n][k].get(metric) is not None for n in names) for k in keys])
            if not use.any():continue
            selected=[k for k,yes in zip(keys,use) if yes];scale=100 if metric in ('add_01','adds_005') else 1
            values=np.asarray([[data[n][k][metric]*scale for n in names] for k in selected])
            w=weights[:,np.isin(clusters,np.unique(sequences[use]))];point,draws=paired_summary(values,objects[use],sequences[use],w)
            stats=dict(frames=int(use.sum()),values=dict(zip(names,point.tolist())),comparisons={})
            for first,second in comparisons:
                i,j=names.index(first),names.index(second);delta=draws[:,i]-draws[:,j];finite=delta[np.isfinite(delta)]
                stats['comparisons'][first+'_vs_'+second]=dict(delta=float(point[i]-point[j]),ci95=np.quantile(finite,[.025,.975]).tolist())
            metrics[metric]=stats
            for name,value in stats['values'].items():exports.append(dict(population=pop,metric=metric,model=name,frames=stats['frames'],value=value))
    activity=[]
    for row in data['smooth_rotation'].values():
        if 'rotation_anchor_coefficient' in row:activity.append(row['rotation_anchor_coefficient'])
    assert activity and np.isfinite(activity).all()
    result['smooth_activity']=dict(frames=len(activity),absolute_mean=float(np.mean(np.abs(activity))),
        positive_fraction=float(np.mean(np.asarray(activity)>0)),negative_fraction=float(np.mean(np.asarray(activity)<0)))
    # Fixed screening conditions; never select or tune on the already completed test.
    def change(pop,metric,comparison):return result['populations'][pop]['metrics'][metric]['comparisons'][comparison]
    checks={}
    for arm in ('control','smooth_rotation'):
        contrast=arm+'_vs_residual';checks[arm]=dict(
            severe_strict_ci_positive=change('visibility_lt_03','adds_005',contrast)['ci95'][0]>0,
            overall_add_point_not_lower=change('all','add_01',contrast)['delta']>=0,
            overall_strict_point_not_lower=change('all','adds_005',contrast)['delta']>=0)
        if result['populations']['bad_initial_first8']['frames']:
            for metric in ('center_mm','rotation_deg'):
                checks[arm]['bad_initial_first8_'+metric+'_point_not_worse']=change('bad_initial_first8',metric,contrast)['delta']<=0
    structural=change('visibility_lt_03','adds_005','smooth_rotation_vs_control')['ci95'][0]>0
    checks['smooth_rotation']['paired_structural_gain']=structural
    checks['smooth_rotation']['read_active']=result['smooth_activity']['absolute_mean']>0
    for metric in ('add_01','adds_005'):
        checks['smooth_rotation']['matched_control_'+metric+'_point_not_lower']=change('all',metric,'smooth_rotation_vs_control')['delta']>=0
    if result['populations']['bad_initial_first8']['frames']:
        for metric in ('center_mm','rotation_deg'):
            checks['smooth_rotation']['matched_control_first8_'+metric+'_point_not_worse']=change('bad_initial_first8',metric,'smooth_rotation_vs_control')['delta']<=0
    eligible=[name for name,value in checks.items() if all(value.values())]
    selected=max(eligible,key=lambda n:result['populations']['visibility_lt_03']['metrics']['adds_005']['values'][n]) if eligible else 'residual'
    result['selection']=dict(selected=selected,new_candidate_promoted=bool(eligible),checks=checks,
        limitation='Conservative point guards are not statistical noninferiority tests. Structural promotion additionally requires a positive severe-occlusion paired interval versus the matched control. No automatic official test launch.')
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (a.out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(exports[0]));writer.writeheader();writer.writerows(exports)
    paired=[]
    for key in keys:
        row=reference[key]
        entry={k:row[k] for k in ('stream_id','frame_index','object_id','physical_sequence','visibility','initialization','initial_bad','updates_after_initialization')}
        for name in names:
            for metric in ('add_01','adds_005','center_mm','rotation_deg'):
                entry[name+'_'+metric]=data[name][key][metric]
        paired.append(entry)
    with (a.out/'paired_frames.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    lines=['# Real non-GT initialization: full s0 val','',result['scope'],'',
        '| Population / strict ADD-S (%) | Frames | '+' | '.join(names)+' |','|---|---:|'+'---:|'*len(names)]
    for name,stats in result['populations'].items():
        if stats['frames']:lines.append(f"| {name} | {stats['frames']} | "+' | '.join(f"{stats['metrics']['adds_005']['values'][n]:.3f}" for n in names)+' |')
    lines+=['',f'Selected by the frozen val rule: {selected}.',result['selection']['limitation']]
    (a.out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
