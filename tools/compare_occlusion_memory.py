"""Compare the completed spatial-memory x temporal-occlusion train/val study."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from compare_rk_ablation import paired_summary,episode_populations,METRICS


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args();r=a.experiment
    protocol=json.loads((r/'augmentation_protocol.json').read_text());old=Path(protocol['unaugmented_experiment'])
    experiments=[json.loads((root/'experiment.json').read_text()) for root in (old,r)]
    for root in (old,r):assert json.loads((root/'status.json').read_text())['phase']=='completed'
    for key in ('parent_sha256','training_manifest_sha256','initial_poses_sha256','split_hash','mesh_hash','seed','steps'):
        assert experiments[0][key]==experiments[1][key],key
    for arm in ('control','spatial'):
        configs=[json.loads((root/arm/'train/config.json').read_text()) for root in (old,r)]
        # config.json is the resolved training config, including explicit world size.
        clean=[{k:v for k,v in c.items() if k not in ('preflight_receipt','temporal_occlusion_probability')} for c in configs]
        assert clean[0]==clean[1],('Non-augmentation config mismatch',arm)
        assert configs[0].get('temporal_occlusion_probability',0)==0 and configs[1]['temporal_occlusion_probability']==.5
    probe=json.loads((r/'occlusion_probe.json').read_text());assert probe['passed'] and probe['no_augmentation_missing_vs_zero_bitwise']
    folders={'parent':Path(experiments[0]['reference_evaluation']),'M0A0':old/'control/s0_val','M1A0':old/'spatial/s0_val','M0A1':r/'control/s0_val','M1A1':r/'spatial/s0_val'}
    data={};manifests={}
    for name,folder in folders.items():
        m=json.loads((folder/'manifest.json').read_text());assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
        assert m['initial_poses_sha256']==experiments[0]['initial_poses_sha256'] and m['fp_calls']==0 and m['split']=='val'
        assert m['split_hash']==experiments[0]['split_hash'] and m['mesh_hash']==experiments[0]['mesh_hash']
        data[name]={(x['stream_id'],x['frame_index']):x for x in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()) if not x['initialization']};manifests[name]=m
    assert manifests['parent']['checkpoint_sha256']==experiments[0]['parent_sha256']
    for names,experiment in [(('M0A0','M1A0'),experiments[0]),(('M0A1','M1A1'),experiments[1])]:
        assert all(manifests[name]['source_sha256']==experiment['source_sha256'] for name in names)
    keys=sorted(data['parent']);assert len(keys)==22880 and all(set(v)==set(keys) for v in data.values())
    assert all(data[name][key]['visibility']==data['parent'][key]['visibility'] for name in data for key in keys)
    names=list(data);reference=[data['parent'][k] for k in keys];seq=np.array(['/'.join(k[0].split('/')[:2]) for k in keys]);obj=np.array([x['object_id'] for x in reference]);clusters=np.unique(seq)
    weights=np.random.default_rng(20260914).multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=2000)
    long,_=episode_populations(reference)
    populations={'all':np.ones(len(keys),bool),'visibility_lt_05':np.array([x['visibility'] is not None and x['visibility']<.5 for x in reference]),
        'visibility_lt_03':np.array([x['visibility'] is not None and x['visibility']<.3 for x in reference]),'long_occlusion_gt_8_frames':np.array([k in long for k in keys])}
    contrasts={'augmentation_without_spatial':[0,-1,0,1,0],'augmentation_with_spatial':[0,0,-1,0,1],
        'spatial_without_augmentation':[0,-1,1,0,0],'spatial_with_augmentation':[0,0,0,-1,1],
        'interaction':[0,1,-1,-1,1],'joint_vs_parent':[-1,0,0,0,1],'augmented_control_vs_parent':[-1,0,0,1,0]}
    result=dict(completed=True,frames=23200,tracked_frames=22880,streams=320,
        scope='One seed, controlled noisy-GT-initialized s0 val, zero FP. Sequential development study: unaugmented arms were inspected before augmented training. Not BOP AR or an untouched confirmation test.',
        factors=dict(M='Dense spatial keyframe residual added to R1K1',A='Temporal RGB-D occlusion during training, probability .5; evaluation images unchanged'),
        checkpoint_sha256={n:m['checkpoint_sha256'] for n,m in manifests.items()},source_sha256={n:m['source_sha256'] for n,m in manifests.items()},
        bootstrap=dict(resamples=2000,unit='physical sequence with all camera streams together',seed=20260914),populations={})
    sequence_rows=[]
    for pop,mask in populations.items():
        result['populations'][pop]=dict(frames=int(mask.sum()),metrics={});w=weights[:,np.isin(clusters,np.unique(seq[mask]))]
        for metric in METRICS:
            scale=100 if metric in ('add_01','adds_01','adds_005','lost') else 1
            values=np.array([[data[name][k][metric]*scale for name in names] for k in keys])[mask]
            point,draws=paired_summary(values,obj[mask],seq[mask],w);stats=dict(values=dict(zip(names,point.tolist())),comparisons={})
            for contrast,coeff in contrasts.items():
                delta=draws@coeff;finite=delta[np.isfinite(delta)];stats['comparisons'][contrast]=dict(delta=float(point@coeff),ci95=np.quantile(finite,[.025,.975]).tolist())
            result['populations'][pop]['metrics'][metric]=stats
            for physical in np.unique(seq[mask]):
                selected=seq[mask]==physical;assert len(np.unique(obj[mask][selected]))==1
                sequence_rows.append(dict(population=pop,physical_sequence=physical,object_id=int(obj[mask][selected][0]),frames=int(selected.sum()),metric=metric,**dict(zip(names,values[selected].mean(0).tolist()))))
    (r/'occlusion_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (r/'occlusion_paired_sequences.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(sequence_rows[0]));writer.writeheader();writer.writerows(sequence_rows)
    lines=['# Spatial memory x temporal occlusion training','',result['scope'],'','| Population / metric (%) | R1K1 parent | M0A0 | M1A0 | M0A1 | M1A1 | Interaction, 95% CI |','|---|---:|---:|---:|---:|---:|---:|']
    for pop,rs in result['populations'].items():
        for metric in ('add_01','adds_005'):
            m=rs['metrics'][metric];d=m['comparisons']['interaction'];lo,hi=d['ci95']
            lines.append(f'| {pop} / {metric} | '+' | '.join(f'{v:.3f}' for v in m['values'].values())+f" | {d['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    (r/'occlusion_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
