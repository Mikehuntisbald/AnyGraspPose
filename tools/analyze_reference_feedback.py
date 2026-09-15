"""Read completed reference-actor outputs on pre-existing fixed diagnostic groups."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(rows,field):
    groups=defaultdict(list)
    for row in rows:groups[row['object_id']].append(row)
    return dict(frames=len(rows),objects=len(groups),successes=sum(r[field] for r in rows),
        object_macro_percent=100*sum(sum(r[field] for r in g)/len(g) for g in groups.values())/len(groups) if groups else None)


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('hold-audit','out'):p.add_argument('--'+key,required=True,type=Path)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--experiment',type=Path);choice.add_argument('--evaluation',action='append')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    hold_audit=json.loads((a.hold_audit/'audit.json').read_text());raw=(a.hold_audit/'frames.jsonl').read_bytes();hold=list(map(json.loads,raw.splitlines()))
    comparison=None
    if a.experiment:
        e=json.loads((a.experiment/'experiment.json').read_text());comparison=json.loads((a.experiment/'comparison.json').read_text());assert comparison['completed']
        assert hold_audit['checkpoint_sha256']==e['parent_sha256']
        folders={'parent':Path(e['reference_evaluation']),'control':a.experiment/'control/s0_val','pose_reference':a.experiment/'pose_reference/s0_val'}
    else:
        folders={}
        for item in a.evaluation:
            name,path=item.split('=',1)
            if not name or name in folders:raise ValueError('Distinct evaluation names required')
            folders[name]=Path(path)
        if 'parent' not in folders:raise ValueError('Original parent evaluation required to bind fixed groups')
    data={};hashes={};manifests={}
    for name,folder in folders.items():
        m=json.loads((folder/'manifest.json').read_text());manifests[name]=m
        assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
        assert m['split']=='val' and m['fp_calls']==m['critic_calls']==0
        if comparison:assert m['checkpoint_sha256']==comparison['checkpoints'][name] and m['initial_poses_sha256']==e['initial_poses_sha256']
        rr=(folder/'predictions.jsonl').read_bytes();hashes[name]=hashlib.sha256(rr).hexdigest()
        entries=list(map(json.loads,rr.splitlines()));data[name]={(r['stream_id'],r['frame_index']):r for r in entries}
        assert len(data[name])==len(entries)==23200
    assert hashes['parent']==hold_audit['prediction_sha256']
    assert manifests['parent']['checkpoint_sha256']==hold_audit['checkpoint_sha256']
    assert all(set(rows)==set(data['parent']) for rows in data.values())
    for name,d in data.items():
        assert all(manifests[name][k]==manifests['parent'][k] for k in ('split_hash','mesh_hash','initial_poses_sha256'))
        assert all(d[k]['pose_centered']==r['pose_centered'] for k,r in data['parent'].items() if r['initialization'])
    rows=[]
    for r in hold:
        key=r['stream_id'],r['frame'];row=dict(r)
        for name,d in data.items():
            assert d[key]['visibility']==r['visibility'];row[name+'_success']=bool(d[key]['adds_005']) and d[key]['status']=='ok'
        rows.append(row)
    populations={'all_occluded':rows,'severe':[r for r in rows if r['visibility']<.3],
        'severe_before_motion':[r for r in rows if r['visibility']<.3 and r['initial_stationary_prefix']],
        'severe_after_motion':[r for r in rows if r['visibility']<.3 and not r['initial_stationary_prefix']],
        'no_prior_clear_severe':[r for r in rows if r['visibility']<.3 and r['category']=='no_prior_three_clear_observations'],
        'fixed_parent_static_hold_benefit':[r for r in rows if r['visibility']<.3 and r['initial_stationary_prefix'] and r['hold_initial_success'] and not r['lip_success']],
        'fixed_parent_moving_lip_benefit':[r for r in rows if r['visibility']<.3 and not r['initial_stationary_prefix'] and r['lip_success'] and not r['hold_initial_success']]}
    summary={}
    for name,rr in populations.items():
        stats={arm:summarize(rr,arm+'_success') for arm in data};stats['initializer_hold']=summarize(rr,'hold_initial_success')
        by_arm={}
        for arm in data:
            coefficients={}
            for field in ('reference_rotation_coefficient','reference_center_coefficient'):
                values=[data[arm][(r['stream_id'],r['frame'])].get(field) for r in rr]
                finite=np.array([v for v in values if v is not None and np.isfinite(v)])
                coefficients[field]=dict(available=len(finite),missing=len(values)-len(finite),mean=float(finite.mean()) if len(finite) else None,
                    quantiles_05_50_95=np.quantile(finite,[.05,.5,.95]).tolist() if len(finite) else None,
                    negative_fraction=float((finite<0).mean()) if len(finite) else None)
            by_arm[arm]=coefficients
        summary[name]=dict(metrics=stats,coefficients=by_arm.get('pose_reference'),coefficients_by_arm=by_arm)
    onsets={}
    for name,d in data.items():
        streams=defaultdict(list)
        for row in d.values():streams[row['stream_id']].append(row)
        failures=0;accurate=0
        for sid,rr in streams.items():
            rr=sorted(rr,key=lambda r:r['frame_index'])
            assert rr[0]['initialization'] and not any(r['initialization'] for r in rr[1:])
            if not rr[0]['adds_005']:continue
            accurate+=1;failures+=any(not r['adds_005'] or r['status']!='ok' for r in rr[1:9])
        onsets[name]=dict(initially_accurate_streams=accurate,failed_within_first8=failures)
    result=dict(completed=True,scope='Same fixed groups from the earlier M1A1/initializer-hold diagnostic. GT motion and parent error define analysis groups only. Point estimates and coefficient distributions are descriptive, not deployed switches or causal mediation evidence.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hold_frames_sha256=hashlib.sha256(raw).hexdigest(),prediction_sha256=hashes,checkpoints={n:m['checkpoint_sha256'] for n,m in manifests.items()},populations=summary,initial_failure=onsets)
    names=[*data,'initializer_hold']
    (a.out/'analysis.json').write_text(json.dumps(result,indent=2));lines=['# Retained pose feedback: fixed-group diagnostics','',result['scope'],'',
        '| Group | Frames | '+' | '.join(names)+' |','|---|---:|'+ '|'.join('---:' for _ in names)+'|']
    for name,stats in summary.items():
        metrics=stats['metrics'];lines.append(f"| {name} | {metrics['parent']['frames']} | "+' | '.join('NA' if metrics[arm]['object_macro_percent'] is None else f"{metrics[arm]['object_macro_percent']:.3f}" for arm in names)+' |')
    (a.out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));print(json.dumps(onsets))


if __name__=='__main__':main()
