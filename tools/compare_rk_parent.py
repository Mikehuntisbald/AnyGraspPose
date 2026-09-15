"""Paired comparison to the untouched selected parent under identical val poses."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_rk_ablation import ARMS,METRICS,paired_summary,episode_populations


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args();root=a.experiment
    folders={'parent':root/'parent_s0_val',**{arm:root/arm/'s0_val' for arm in ARMS}};data={};manifests={}
    for name,folder in folders.items():
        m=json.loads((folder/'manifest.json').read_text());assert m['completed'] and m['population_verified'] and m['frames']==23200
        manifests[name]=m;data[name]={(r['stream_id'],r['frame_index']):r for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()) if not r['initialization']}
    base=manifests['parent'];keys=sorted(data['parent']);names=list(folders)
    for name,m in manifests.items():
        assert set(data[name])==set(keys)
        for field in ('initial_poses_sha256','split_hash','mesh_hash','source_sha256','split'):assert m[field]==base[field]
        assert m['fp_calls']==0
    reference=[data['parent'][key] for key in keys];sequences=np.array(['/'.join(k[0].split('/')[:2]) for k in keys]);objects=np.array([r['object_id'] for r in reference])
    clusters=np.unique(sequences);weights=np.random.default_rng(20260913).multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=2000)
    long_keys,_=episode_populations(reference)
    populations={'all':np.ones(len(keys),bool),'visibility_lt_05':np.array([r['visibility'] is not None and r['visibility']<.5 for r in reference]),
        'visibility_lt_03':np.array([r['visibility'] is not None and r['visibility']<.3 for r in reference]),
        'visibility_ge_05':np.array([r['visibility'] is not None and r['visibility']>=.5 for r in reference]),
        'long_occlusion_gt_8_frames':np.array([key in long_keys for key in keys])}
    report=dict(completed=True,frames=23200,tracked_frames=len(keys),initialization=base['initial_pose_source'],initial_poses_sha256=base['initial_poses_sha256'],
        estimator='Object macro, paired physical-sequence bootstrap, 2000 shared draws, seed 20260913; one training seed',
        checkpoints={name:m['checkpoint_sha256'] for name,m in manifests.items()},populations={})
    for pop,mask in populations.items():
        target=dict(frames=int(mask.sum()),metrics={});report['populations'][pop]=target
        if not mask.any():continue
        present=np.isin(clusters,np.unique(sequences[mask]))
        for metric in METRICS:
            scale=100 if metric in ('add_01','adds_01','adds_005','lost') else 1
            values=np.array([[data[name][key][metric]*scale for name in names] for key in keys])[mask]
            point,draws=paired_summary(values,objects[mask],sequences[mask],weights[:,present])
            result=dict(values=dict(zip(names,point.tolist())),versus_parent={})
            for i,arm in enumerate(ARMS,1):
                delta=draws[:,i]-draws[:,0];delta=delta[np.isfinite(delta)]
                result['versus_parent'][arm]=dict(delta=float(point[i]-point[0]),ci95=np.quantile(delta,[.025,.975]).tolist(),valid_draws=len(delta))
            target['metrics'][metric]=result
    (root/'parent_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Matched validation vs the selected parent','',''+report['initialization'],
        '', '| Population / metric (%) | Parent | R0K0 | R1K0 | R0K1 | R1K1 | Joint vs parent (95% CI) |','|---|---:|---:|---:|---:|---:|---:|']
    for pop,rs in report['populations'].items():
        for metric in ('add_01','adds_005'):
            m=rs['metrics'].get(metric)
            if not m:continue
            delta=m['versus_parent']['R1K1'];lo,hi=delta['ci95']
            lines.append(f'| {pop} / {metric} | '+' | '.join(f'{v:.3f}' for v in m['values'].values())+f" | {delta['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines+=['','This controlled validation is not a non-GT initialization or official BOP-AR claim. Selection uses validation only.']
    (root/'parent_comparison.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
