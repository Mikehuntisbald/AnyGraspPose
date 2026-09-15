"""Read-only event diagnostics with fixed reference entry states and censoring."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def visibility_class(row):
    v=row['visibility']
    return 'unknown' if v is None else ('low' if v<.5 else 'clear')


def extract_episodes(rows):
    groups={};seen=set();events=[]
    for row in rows:
        key=(row['stream_id'],row['frame_index'])
        if key in seen:raise ValueError('Duplicate frame identity')
        seen.add(key);groups.setdefault(row['stream_id'],[]).append(row)
    for sid,rs in sorted(groups.items()):
        rs=sorted(rs,key=lambda r:r['frame_index']);i=0
        while i<len(rs):
            if rs[i]['initialization'] or visibility_class(rs[i])!='low':i+=1;continue
            j=i+1
            while j<len(rs) and rs[j]['frame_index']==rs[j-1]['frame_index']+1 and visibility_class(rs[j])=='low':j+=1
            pre=[];following=rs[i]['frame_index']
            for index in range(i-1,max(-1,i-4),-1):
                row=rs[index]
                if row['initialization'] or row['frame_index']!=following-1 or visibility_class(row)!='clear':break
                pre.insert(0,row['frame_index']);following=row['frame_index']
            post=[];previous=rs[j-1]['frame_index'];reason=None
            for index in range(j,min(len(rs),j+10)):
                row=rs[index]
                if row['frame_index']!=previous+1:reason='frame_gap';break
                if visibility_class(row)!='clear':reason='unknown_visibility' if row['visibility'] is None else 'next_occlusion';break
                post.append(row['frame_index']);previous=row['frame_index']
            if len(post)<10 and reason is None:reason='stream_end'
            events.append(dict(stream_id=sid,object_id=rs[i]['object_id'],start=rs[i]['frame_index'],end=rs[j-1]['frame_index'],
                frames=[x['frame_index'] for x in rs[i:j]],length=j-i,pre_frames=pre,pre_complete=len(pre)==3,
                post_frames=post,post_complete=len(post)==10,censor_reason=reason))
            i=j
    return events


def successful(row,metric='adds_005'):
    return bool(row[metric]) and row['status']=='ok' and not row['initialization']


def stable_recovery(rows,metric='adds_005',consecutive=3):
    """Return the third successful clear frame's 1-based offset, never a single hit."""
    run=0;previous=None
    for offset,row in enumerate(rows,1):
        if previous is not None and row['frame_index']!=previous+1:run=0
        run=run+1 if successful(row,metric) and visibility_class(row)=='clear' else 0;previous=row['frame_index']
        if run>=consecutive:return offset
    return None


def main():
    import numpy as np
    from compare_rk_ablation import paired_summary
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',action='append',required=True,help='name=folder');p.add_argument('--reference',default='parent')
    p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    folders=dict(x.split('=',1) for x in a.evaluation);assert len(folders)==len(a.evaluation) and a.reference in folders
    data={};manifests={};hashes={}
    for name,path in folders.items():
        folder=Path(path);m=json.loads((folder/'manifest.json').read_text());assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320 and m['split']=='val'
        assert m['fp_calls']==m['critic_calls']==0
        raw=(folder/'predictions.jsonl').read_bytes();rows=list(map(json.loads,raw.splitlines()));d={(x['stream_id'],x['frame_index']):x for x in rows}
        assert len(d)==23200;data[name]=d;manifests[name]=m;hashes[name]=hashlib.sha256(raw).hexdigest()
    ref=data[a.reference];rm=manifests[a.reference]
    for name,m in manifests.items():
        assert all(m[k]==rm[k] for k in ('split_hash','mesh_hash','initial_poses_sha256','initialization_uses_gt_pose'))
        assert set(data[name])==set(ref)
        assert all(data[name][k]['visibility']==ref[k]['visibility'] for k in ref)
    events=extract_episodes(list(ref.values()));names=list(data)
    for e in events:
        sid=e['stream_id'];pre=[ref[(sid,f)] for f in e['pre_frames']]
        e['reference_entry']='unknown' if not e['pre_complete'] else ('stable_good' if all(successful(x) for x in pre) else 'not_stable_good')
        e['reference_entry_adds01']='unknown' if not e['pre_complete'] else ('stable_good' if all(successful(x,'adds_01') for x in pre) else 'not_stable_good')
        e['arms']={}
        for name,rows in data.items():
            occluded=[rows[(sid,f)] for f in e['frames']];after=[rows[(sid,f)] for f in e['post_frames']];end_ok=successful(occluded[-1])
            recovery=stable_recovery(after)
            e['arms'][name]=dict(occluded_success_frames=sum(successful(x) for x in occluded),
                failed_any=any(not successful(x) for x in occluded),end_success=end_ok,
                stable_recovery_offset=recovery,complete_window_recovery=(recovery is not None) if e['post_complete'] and not end_ok else None,
                mean_center_mm=float(np.mean([x['center_mm'] for x in occluded])),max_center_mm=max(x['center_mm'] for x in occluded),
                mean_rotation_deg=float(np.mean([x['rotation_deg'] for x in occluded])))
    groups={'all':events,'long_gt8':[e for e in events if e['length']>8],
        'reference_stable_good_entry':[e for e in events if e['reference_entry']=='stable_good'],
        'reference_not_stable_good_entry':[e for e in events if e['reference_entry']=='not_stable_good'],
        'reference_unknown_entry':[e for e in events if e['reference_entry']=='unknown'],
        'long_gt8_reference_stable_good_entry':[e for e in events if e['length']>8 and e['reference_entry']=='stable_good']}
    result=dict(completed=True,scope='Read-only development s0 val diagnosis, fixed reference entry population for every arm. Controlled noisy-GT initialization, zero FP; not a new tracker, causal intervention or official AR.',
        reference=a.reference,checkpoint_sha256={n:m['checkpoint_sha256'] for n,m in manifests.items()},prediction_sha256=hashes,
        definitions=dict(occlusion='visibility proxy <.5; gaps and unknown values split episodes',stable_entry='three immediately preceding non-initialization clear frames all ADD-S<.05d and status ok in frozen reference',
            recovery='three consecutive ADD-S<.05d status-ok clear frames; 1-based offset of third frame; rate denominator requires full 10-frame clear window and failure at occlusion end',
            intervals='2000 physical-sequence paired bootstrap draws; object-macro frame metrics; one training seed'),populations={},events=events)
    allkeys=[(e['stream_id'],f) for e in events for f in e['frames']];allseq=np.unique(['/'.join(k[0].split('/')[:2]) for k in allkeys])
    weights=np.random.default_rng(20260914).multinomial(len(allseq),np.ones(len(allseq))/len(allseq),size=2000)
    for group,es in groups.items():
        keys=[(e['stream_id'],f) for e in es for f in e['frames']];seq=np.array(['/'.join(k[0].split('/')[:2]) for k in keys]);obj=np.array([ref[k]['object_id'] for k in keys])
        r=dict(episodes=len(es),physical_sequences=len(np.unique(seq)),frames=len(keys),complete_post_windows=sum(e['post_complete'] for e in es),arms={},metrics={});result['populations'][group]=r
        if not keys:continue
        for name in names:
            failed=[e for e in es if e['post_complete'] and not e['arms'][name]['end_success']]
            r['arms'][name]=dict(episodes_with_any_failure=sum(e['arms'][name]['failed_any'] for e in es),failed_at_end=sum(not e['arms'][name]['end_success'] for e in es),
                recovery_denominator=len(failed),recovered_stably=sum(e['arms'][name]['complete_window_recovery'] for e in failed),
                frame_micro_success=sum(successful(data[name][k]) for k in keys)/len(keys))
        for metric in ('add_01','adds_005','center_mm','rotation_deg'):
            values=np.array([[data[n][k][metric]*(100 if metric in ('add_01','adds_005') else 1) for n in names] for k in keys])
            point,draws=paired_summary(values,obj,seq,weights[:,np.isin(allseq,np.unique(seq))]);comparisons={};ri=names.index(a.reference)
            for i,name in enumerate(names):
                if name==a.reference:continue
                delta=draws[:,i]-draws[:,ri];finite=delta[np.isfinite(delta)];comparisons[name+'_vs_'+a.reference]=dict(delta=float(point[i]-point[ri]),ci95=np.quantile(finite,[.025,.975]).tolist())
            r['metrics'][metric]=dict(values=dict(zip(names,point.tolist())),comparisons=comparisons)
    (a.out/'events.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    flat=[]
    for e in events:
        for name,v in e['arms'].items():flat.append(dict(stream_id=e['stream_id'],object_id=e['object_id'],start=e['start'],end=e['end'],length=e['length'],reference_entry=e['reference_entry'],post_complete=e['post_complete'],censor_reason=e['censor_reason'],arm=name,**v))
    with (a.out/'paired_events.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# Occlusion entry and recovery diagnosis','',result['scope'],'','Stable entry requires three accurate clear reference frames. Accuracy below is object-macro ADD-S@0.05d. Recovery denominators remain explicit.','',
        '| Population | Episodes | Frames | '+' | '.join(names)+' |','|---|---:|---:|'+'---:|'*len(names)]
    for name,r in result['populations'].items():
        if not r['frames']:continue
        lines.append(f"| {name} | {r['episodes']} | {r['frames']} | "+' | '.join(f'{v:.3f}%' for v in r['metrics']['adds_005']['values'].values())+' |')
    lines+=['','Entry groups are defined by the frozen reference, not by each tested arm. Unknown entry and censored recovery episodes are retained in the data. No failure-rate ratio with a changed denominator is presented as a causal recovery gain.']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
