"""Paired object and initializer-quality diagnostics on completed controlled val."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_rk_ablation import paired_summary


def initial_groups(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['stream_id']].append(row)
    initial={}
    for sid,rs in grouped.items():
        ordered=sorted(rs,key=lambda r:r['frame_index'])
        if not ordered[0]['initialization'] or any(r['initialization'] for r in ordered[1:]):
            raise ValueError('Exactly one first initialization record required per stream')
        initial[sid]=dict(good=bool(ordered[0]['adds_005']),very_bad=not bool(ordered[0]['adds_01']),
            object_id=ordered[0]['object_id'],frame=ordered[0]['frame_index'])
    return initial


def stable_success_position(rows):
    """Position of third consecutive accurate usable update; initialization excluded."""
    run=0;previous=None;position=0
    for row in sorted(rows,key=lambda r:r['frame_index']):
        if row['initialization']:continue
        position+=1;good=bool(row['adds_005']) and row['status']=='ok'
        run=(run+1 if previous is not None and row['frame_index']==previous+1 else 1) if good else 0
        if run>=3:return position
        previous=row['frame_index']
    return None


def effect(values,objects,sequences,weights,reference_index):
    point,boot=paired_summary(values,objects,sequences,weights)
    differences=boot-boot[:,reference_index:reference_index+1];result=[]
    for i in range(values.shape[1]):
        finite=differences[:,i][np.isfinite(differences[:,i])]
        result.append(dict(value=float(point[i]),delta=float(point[i]-point[reference_index]),
            ci95=np.quantile(finite,[.025,.975]).tolist() if len(np.unique(sequences))>=2 and len(finite) else None,
            bootstrap_finite_draws=len(finite)))
    return result


def analyze(folders,reference,out):
    if reference not in folders:raise ValueError('Reference evaluation must be supplied')
    out.mkdir(parents=True,exist_ok=False);data={};manifests={};hashes={}
    for name,folder in folders.items():
        m=json.loads((folder/'manifest.json').read_text())
        assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
        assert m['split']=='val' and m['fp_calls']==m['critic_calls']==0
        raw=(folder/'predictions.jsonl').read_bytes();rows=list(map(json.loads,raw.splitlines()))
        data[name]={(r['stream_id'],r['frame_index']):r for r in rows};assert len(data[name])==len(rows)==23200
        manifests[name]=m;hashes[name]=hashlib.sha256(raw).hexdigest()
    ref=data[reference];base=manifests[reference]
    for name,rows in data.items():
        assert set(rows)==set(ref)
        assert all(manifests[name][k]==base[k] for k in ('split_hash','mesh_hash','initial_poses_sha256'))
        assert all(rows[k]['visibility']==r['visibility'] and rows[k]['object_id']==r['object_id'] for k,r in ref.items())
        assert all(rows[k]['pose_centered']==r['pose_centered'] for k,r in ref.items() if r['initialization'])
    names=list(folders);ri=names.index(reference);initial=initial_groups(list(ref.values()))
    keys=sorted(k for k,r in ref.items() if not r['initialization']);rows=[ref[k] for k in keys]
    objects=np.array([r['object_id'] for r in rows]);sequences=np.array(['/'.join(k[0].split('/')[:2]) for k in keys]);clusters=np.unique(sequences)
    weights=np.random.default_rng(20260914).multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=2000)
    low=np.array([r['visibility'] is not None and r['visibility']<.5 for r in rows]);severe=np.array([r['visibility'] is not None and r['visibility']<.3 for r in rows])
    good=np.array([initial[k[0]]['good'] for k in keys]);very_bad=np.array([initial[k[0]]['very_bad'] for k in keys])
    populations={'all':np.ones(len(keys),bool),'visibility_lt05':low,'visibility_lt03':severe,
        'initial_good':good,'initial_bad':~good,'initial_very_bad':very_bad,
        'initial_good_visibility_lt05':good&low,'initial_bad_visibility_lt05':~good&low,
        'initial_good_visibility_lt03':good&severe,'initial_bad_visibility_lt03':~good&severe}
    report=dict(completed=True,reference=reference,names=names,checkpoint_sha256={n:m['checkpoint_sha256'] for n,m in manifests.items()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prediction_sha256=hashes,initial_poses_sha256=base['initial_poses_sha256'],
        scope='Exploratory completed controlled-val diagnostics, one training seed, zero FP. Initial-quality groups use only the shared initialization GT error; they are not available deployment gates. Each group has its own object set. Per-object and leave-one-object-out values describe concentration, not an independent replication.',
        bootstrap=dict(draws=2000,seed=20260914,physical_sequences=len(clusters),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
            limitation='Cameras cluster together. Intervals with fewer than two physical sequences are omitted. Per-object intervals are exploratory and not multiplicity-adjusted.'),populations={})
    object_rows=[]
    for population,mask in populations.items():
        if not mask.any():report['populations'][population]=dict(frames=0);continue
        present_objects=np.unique(objects[mask]);stats=dict(frames=int(mask.sum()),objects=len(present_objects),physical_sequences=len(np.unique(sequences[mask])),metrics={})
        for metric in ('add_01','adds_005'):
            values=np.array([[data[n][k][metric]*100 for n in names] for k in keys]);subweights=weights[:,np.isin(clusters,np.unique(sequences[mask]))]
            summary=effect(values[mask],objects[mask],sequences[mask],subweights,ri)
            stats['metrics'][metric]=dict(zip(names,summary));deltas={n:[] for n in names}
            for oid in present_objects:
                selected=mask&(objects==oid);w=weights[:,np.isin(clusters,np.unique(sequences[selected]))]
                individual=effect(values[selected],objects[selected],sequences[selected],w,ri)
                for name,value in zip(names,individual):
                    deltas[name].append(value['delta']);ci=value['ci95']
                    object_rows.append(dict(population=population,metric=metric,object_id=int(oid),arm=name,frames=int(selected.sum()),
                        physical_sequences=len(np.unique(sequences[selected])),value=value['value'],delta=value['delta'],
                        macro_contribution=value['delta']/len(present_objects),ci_low=None if ci is None else ci[0],ci_high=None if ci is None else ci[1]))
            for name,d in deltas.items():
                value=stats['metrics'][metric][name];assert abs(np.mean(d)-value['delta'])<1e-9
                loo=[(sum(d)-x)/(len(d)-1) for x in d] if len(d)>1 else []
                value['object_influence']=dict(positive=sum(x>1e-10 for x in d),negative=sum(x< -1e-10 for x in d),unchanged=sum(abs(x)<=1e-10 for x in d),
                    leave_one_object_out_delta_range=[min(loo),max(loo)] if loo else None)
        report['populations'][population]=stats
    recovery=[]
    for name,d in data.items():
        grouped=defaultdict(list)
        for row in d.values():grouped[row['stream_id']].append(row)
        for sid,rr in grouped.items():
            if initial[sid]['good']:continue
            pos=stable_success_position(rr);n=len(rr)-1
            recovery.append(dict(arm=name,stream_id=sid,physical_sequence='/'.join(sid.split('/')[:2]),object_id=initial[sid]['object_id'],
                observed_updates=n,third_consecutive_success_position=pos,**{f'by{h}':(pos is not None and pos<=h) if n>=h else None for h in (8,16,32)},by_end=pos is not None))
    report['bad_initial_recovery']={}
    for name in names:
        rr=[r for r in recovery if r['arm']==name]
        report['bad_initial_recovery'][name]=dict(streams=len(rr),**{f'by{h}':dict(eligible=sum(r[f'by{h}'] is not None for r in rr),recovered=sum(r[f'by{h}'] is True for r in rr)) for h in (8,16,32)},by_end=sum(r['by_end'] for r in rr))
    for filename,rr in [('per_object.csv',object_rows),('bad_initializer_streams.csv',recovery)]:
        with (out/filename).open('w') as f:
            fields=list(rr[0]) if rr else ['arm','stream_id','physical_sequence','object_id','observed_updates','third_consecutive_success_position','by8','by16','by32','by_end']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rr)
    report['csv_sha256']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('per_object.csv','bad_initializer_streams.csv')}
    (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Object concentration and initializer recovery','',report['scope'],'',
        '| Population / strict ADD-S (%) | Frames | Objects | '+' | '.join(names)+' |','|---|---:|---:|'+'---:|'*len(names)]
    for population,stats in report['populations'].items():
        if stats['frames']:lines.append(f"| {population} | {stats['frames']} | {stats['objects']} | "+' | '.join(f"{stats['metrics']['adds_005'][n]['value']:.3f}" for n in names)+' |')
    (out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',action='append',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    pairs=[x.split('=',1) for x in a.evaluation]
    if len(pairs)<2 or len({x[0] for x in pairs})!=len(pairs):p.error('Two or more uniquely named evaluations required')
    analyze({n:Path(folder) for n,folder in pairs},a.reference,a.out)
