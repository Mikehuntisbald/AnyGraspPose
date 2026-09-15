"""Full-val startup 2x2 contrasts, frozen-parent controls and paired sequence sums."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import hashlib
import numpy as np
from prepare_startup_factorial import ARMS
from compare_rk_ablation import paired_summary,episode_populations,METRICS


def contrasts(include_previous=False):
    # Column order: frozen parent, S0O0, S0O1, S1O0, S1O1.
    result=dict(S_main=[0,-.5,-.5,.5,.5],O_main=[0,-.5,.5,-.5,.5],interaction=[0,1,-1,-1,1])
    for i,arm in enumerate(ARMS,1):
        c=[-1,0,0,0,0];c[i]=1;result[arm+'_vs_parent']=c
        if i>1:
            c=[0,-1,0,0,0];c[i]=1;result[arm+'_vs_S0O0']=c
    if include_previous:
        result={k:v+[0] for k,v in result.items()}
        for i,arm in enumerate(ARMS,1):
            c=[0]*6;c[i]=1;c[-1]=-1;result[arm+'_vs_previous_candidate']=c
    return result


def compare(root):
    root=Path(root);e=json.loads((root/'experiment.json').read_text())
    folders={'parent':Path(e['reference_evaluation']),**{arm:root/arm/'s0_val' for arm in ARMS}}
    previous=e.get('previous_candidate')
    if previous:folders['previous_candidate']=Path(previous['evaluation'])
    data={};manifests={};hashes={}
    for name,folder in folders.items():
        m=json.loads((folder/'manifest.json').read_text())
        assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320 and m['split']=='val'
        assert m['fp_calls']==0 and all(m[k]==e[k] for k in ('split_hash','mesh_hash','initial_poses_sha256'))
        if name=='parent':assert m['checkpoint_sha256']==e['parent_sha256']
        elif name in ARMS:assert m['source_sha256']==e['source_sha256']
        else:assert m['checkpoint_sha256']==previous['checkpoint_sha256']
        raw=(folder/'predictions.jsonl').read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
        data[name]={(r['stream_id'],r['frame_index']):r for r in map(json.loads,raw.splitlines()) if not r['initialization']};manifests[name]=m
    keys=sorted(data['parent']);names=list(folders);assert all(set(rows)==set(keys) for rows in data.values())
    assert all(data[name][k]['visibility']==data['parent'][k]['visibility'] for name in names for k in keys)
    ref=[data['parent'][k] for k in keys];obj=np.array([r['object_id'] for r in ref]);seq=np.array(['/'.join(k[0].split('/')[:2]) for k in keys])
    clusters=np.unique(seq);weights=np.random.default_rng(20260914).multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=2000)
    long,_=episode_populations(ref)
    masks={'all':np.ones(len(keys),bool),'visibility_lt_05':np.array([r['visibility'] is not None and r['visibility']<.5 for r in ref]),
        'visibility_lt_03':np.array([r['visibility'] is not None and r['visibility']<.3 for r in ref]),'long_occlusion_gt_8_frames':np.array([k in long for k in keys])}
    result=dict(completed=True,frames=23200,tracked_frames=len(keys),streams=320,
        scope='Controlled noisy-GT initialization, zero FP; exploratory one-seed development val. Object-macro paired bootstrap over physical sequences, cameras together. Training observations and label counts match; S1 has additional startup backward compute. Not official AR or seed robustness.',
        checkpoints={n:m['checkpoint_sha256'] for n,m in manifests.items()},prediction_sha256=hashes,
        factors=e['factors'],populations={});export=[]
    for pop,mask in masks.items():
        selected=[k for k,m in zip(keys,mask) if m];stats=dict(frames=len(selected),metrics={})
        w=weights[:,np.isin(clusters,np.unique(seq[mask]))]
        for metric in METRICS:
            scale=100 if metric in ('add_01','adds_01','adds_005','lost') else 1
            values=np.array([[data[n][k][metric]*scale for n in names] for k in keys])[mask]
            point,draws=paired_summary(values,obj[mask],seq[mask],w)
            cstats={}
            for name,coefficient in contrasts(bool(previous)).items():
                delta=draws@coefficient;finite=delta[np.isfinite(delta)]
                cstats[name]=dict(delta=float(point@coefficient),ci95=np.quantile(finite,[.025,.975]).tolist())
            stats['metrics'][metric]=dict(values=dict(zip(names,point.tolist())),contrasts=cstats)
        result['populations'][pop]=stats
        metrics=('add_01','adds_005','center_mm','rotation_deg')
        for arm in names:
            grouped=defaultdict(lambda:dict(frames=0,**{m:0. for m in metrics}))
            for key in selected:
                row=data[arm][key];g=grouped[('/'.join(key[0].split('/')[:2]),row['object_id'])];g['frames']+=1
                for metric in metrics:g[metric]+=row[metric]*(100 if metric in ('add_01','adds_005') else 1)
            per_object=defaultdict(lambda:dict(frames=0,**{m:0. for m in metrics}))
            for (physical,oid),totals in grouped.items():
                export.append(dict(population=pop,arm=arm,physical_sequence=physical,object_id=oid,**totals))
                for field,value in totals.items():per_object[oid][field]+=value
            for metric in metrics:
                value=sum(g[metric]/g['frames'] for g in per_object.values())/len(per_object)
                assert abs(value-stats['metrics'][metric]['values'][arm])<1e-9
    with (root/'paired_physical_sequences.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
    result['paired_csv']=dict(rows=len(export),sha256=hashlib.sha256((root/'paired_physical_sequences.csv').read_bytes()).hexdigest(),four_metrics_reaggregate=True)
    (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['# Startup supervision x startup occlusion','',result['scope'],'',
        '| Population / metric (%) | '+' | '.join(names)+' | Interaction, 95% CI |','|---|'+'---:|'*(len(names)+1)]
    for pop,stats in result['populations'].items():
        for metric in ('add_01','adds_005'):
            v=stats['metrics'][metric];c=v['contrasts']['interaction']
            lines.append(f'| {pop} / {metric} | '+' | '.join(f'{x:.3f}' for x in v['values'].values())+f" | {c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |")
    (root/'report.md').write_text('\n'.join(lines)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True,type=Path);compare(p.parse_args().out)
