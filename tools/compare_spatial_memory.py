"""Paired full-s0-val comparison for dense keyframe memory."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_rk_ablation import paired_summary,episode_populations,METRICS


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args();r=a.experiment;e=json.loads((r/'experiment.json').read_text())
    reader=e.get('reader_arm','spatial');folders={'parent':Path(e['reference_evaluation']),'control':r/'control/s0_val',reader:r/reader/'s0_val'};data={};manifests={}
    for name,folder in folders.items():
        m=json.loads((folder/'manifest.json').read_text());assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
        assert m['initial_poses_sha256']==e['initial_poses_sha256'] and m['fp_calls']==0
        assert m['split_hash']==e['split_hash'] and m['mesh_hash']==e['mesh_hash'] and m['split']=='val'
        manifests[name]=m;data[name]={(x['stream_id'],x['frame_index']):x for x in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()) if not x['initialization']}
    assert manifests['parent']['checkpoint_sha256']==e['parent_sha256']
    assert manifests['control']['source_sha256']==manifests[reader]['source_sha256']==e['source_sha256']
    keys=sorted(data['parent']);names=list(folders)
    assert all(set(v)==set(keys) for v in data.values())
    assert all(data['parent'][key]['visibility']==data[name][key]['visibility'] for key in keys for name in names)
    reference=[data['parent'][k] for k in keys];seq=np.array(['/'.join(k[0].split('/')[:2]) for k in keys]);obj=np.array([x['object_id'] for x in reference]);clusters=np.unique(seq)
    weights=np.random.default_rng(20260914).multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=2000)
    long,_=episode_populations(reference)
    populations={'all':np.ones(len(keys),bool),'visibility_lt_05':np.array([x['visibility'] is not None and x['visibility']<.5 for x in reference]),
        'visibility_lt_03':np.array([x['visibility'] is not None and x['visibility']<.3 for x in reference]),
        'long_occlusion_gt_8_frames':np.array([k in long for k in keys])}
    result=dict(completed=True,frames=23200,tracked_frames=len(keys),streams=320,initialization=manifests['parent']['initial_pose_source'],
        scope='Controlled noisy-GT-initialized val, zero FP, one training seed. Paired object macro with physical-sequence bootstrap; not BOP AR or deployment initialization accuracy.',
        checkpoints={name:m['checkpoint_sha256'] for name,m in manifests.items()},source_sha256={name:m['source_sha256'] for name,m in manifests.items()},populations={})
    for pop,mask in populations.items():
        result['populations'][pop]=dict(frames=int(mask.sum()),metrics={})
        if not mask.any():continue
        w=weights[:,np.isin(clusters,np.unique(seq[mask]))]
        for metric in METRICS:
            scale=100 if metric in ('add_01','adds_01','adds_005','lost') else 1
            values=np.array([[data[name][k][metric]*scale for name in names] for k in keys])[mask]
            point,draws=paired_summary(values,obj[mask],seq[mask],w);stats=dict(values=dict(zip(names,point.tolist())),comparisons={})
            for contrast,coeff in [(reader+'_vs_control',[0,-1,1]),(reader+'_vs_parent',[-1,0,1]),('control_vs_parent',[-1,1,0])]:
                delta=draws@coeff;finite=delta[np.isfinite(delta)];stats['comparisons'][contrast]=dict(delta=float(point@coeff),ci95=np.quantile(finite,[.025,.975]).tolist())
            result['populations'][pop]['metrics'][metric]=stats
    (r/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    title={'smooth_rotation':'Smooth signed rotation anchor','rotation_anchor':'Retained rotation anchor','adaptive_reference':'Adaptive pose reference','natural':'Natural occlusion sampling','direct_pose':'Direct pose residual','pose_reference':'Retained pose feedback'}.get(reader,reader.capitalize()+' keyframe memory')
    lines=['# '+title+': paired validation','',''+result['scope'],'',f'| Population / metric (%) | Parent | Control | {reader.capitalize()} | {reader.capitalize()} vs control, 95% CI |','|---|---:|---:|---:|---:|']
    for pop,rs in result['populations'].items():
        for metric in ('add_01','adds_005'):
            m=rs['metrics'].get(metric)
            if not m:continue
            d=m['comparisons'][reader+'_vs_control'];lo,hi=d['ci95']
            lines.append(f'| {pop} / {metric} | '+' | '.join(f'{v:.3f}' for v in m['values'].values())+f" | {d['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    (r/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
