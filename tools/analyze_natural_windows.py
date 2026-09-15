"""Describe natural occlusion and eligible 8+48 training windows, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def runs(mask):
    padded=np.r_[False,np.asarray(mask,dtype=bool),False].astype('i1');edges=np.diff(padded)
    return list(zip(np.flatnonzero(edges==1).tolist(),np.flatnonzero(edges==-1).tolist()))


def describe_window(values,inside,burn=8,unroll=48):
    v=np.array([np.nan if x is None else x for x in values],dtype='f8');inside=np.asarray(inside,dtype=bool)
    if len(v)!=burn+unroll or inside.shape!=v.shape:raise ValueError('Expected the full observed burn-in and unroll')
    low=v<.5;severe=v<.3;clear=v>=.5;sup=v[burn:];long_low=runs(low);long_severe=runs(severe)
    retention=False;recovery=False
    for start,end in long_low:
        if start<burn or end-start<9:continue
        if not (clear[start-burn:start].all() and severe[start:end].sum()>=4 and inside[start:end].all()):continue
        if end+4<=len(v) and clear[end:end+4].all():retention=True
    for start,end in long_severe:
        if end-max(start,burn)<9 or not inside[start:end].all():continue
        if end+4<=len(v) and clear[end:end+4].all():recovery=True
    return dict(natural_lt05_frames=int((sup<.5).sum()),natural_lt03_frames=int((sup<.3).sum()),unknown_frames=int(np.isnan(sup).sum()),
        longest_lt05_run=max((e-s for s,e in runs(low[burn:])),default=0),longest_lt03_run=max((e-s for s,e in runs(severe[burn:])),default=0),
        longest_inframe_lt03_run=max((e-s for s,e in runs((severe&inside)[burn:])),default=0),retention_ready=bool(retention),recovery_ready=bool(recovery))


def summary(records):
    if not records:return dict(windows=0,supervised_frames=0)
    total=len(records)*48
    return dict(windows=len(records),supervised_frames=total,
        natural_lt05_frames=sum(r['natural_lt05_frames'] for r in records),natural_lt03_frames=sum(r['natural_lt03_frames'] for r in records),unknown_frames=sum(r['unknown_frames'] for r in records),
        natural_lt05_fraction=sum(r['natural_lt05_frames'] for r in records)/total,natural_lt03_fraction=sum(r['natural_lt03_frames'] for r in records)/total,
        long_low_windows=sum(r['longest_lt05_run']>8 for r in records),long_severe_windows=sum(r['longest_lt03_run']>8 for r in records),
        retention_windows=sum(r['retention_ready'] for r in records),recovery_windows=sum(r['recovery_ready'] for r in records))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--cache',required=True,type=Path);p.add_argument('--manifest',action='append',default=[],help='name=training_samples.json');p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);m=json.loads((a.cache/'completed.json').read_text());assert m['completed'] and m['all_frame_positions_covered'] and m['cached_mismatches']==0
    raw=(a.cache/'train_visibility.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==m['output_sha256']
    streams=sorted(map(json.loads,raw.splitlines()),key=lambda r:r['stream_id']);assert len(streams)==m['streams'];windows=[];lookup={};raw_values=[];episode_lengths={'.5':[],'.3':[]}
    for sid,s in enumerate(streams):
        assert len(s['frames'])==len(s['values'])==len(s['bbox_fully_inside_image'])
        v=np.array([np.nan if x is None else x for x in s['values']],dtype='f8');raw_values.extend(v[1:].tolist())
        # Current data is contiguous; refuse to count episodes across unobserved gaps.
        if not np.array_equal(s['frames'],np.arange(s['frames'][0],s['frames'][0]+len(v))):raise ValueError('Noncontiguous stream requires explicit gap-aware episode definitions')
        for threshold,key in [(.5,'.5'),(.3,'.3')]:episode_lengths[key].extend(e-start for start,e in runs(v[1:]<threshold))
        for start in range(len(v)-56):
            d=describe_window(s['values'][start+1:start+57],s['bbox_fully_inside_image'][start+1:start+57])
            d.update(stream=sid,stream_id=s['stream_id'],object_id=s['object_id'],physical_sequence='/'.join(s['stream_id'].split('/')[:2]),start=start,start_frame=s['frames'][start],end_frame=s['frames'][start+56]);windows.append(d);lookup[(sid,start)]=d
    values=np.array(raw_values);report=dict(completed=True,scope='Train-only natural-visibility population and sampling audit. Labels describe raw observations BEFORE synthetic overlays; no pose accuracy or causal training effect is estimated.',
        visibility_cache_sha256=m['output_sha256'],visibility_provenance_limit=m['provenance_limit'],raw_frames_including_initialization=m['frames'],raw_tracked_positions=len(values),
        raw_natural_lt05_frames=int((values<.5).sum()),raw_natural_lt03_frames=int((values<.3).sum()),raw_unknown_frames=int(np.isnan(values).sum()),
        raw_natural_lt05_fraction=float((values<.5).mean()),raw_natural_lt03_fraction=float((values<.3).mean()),
        episodes={k:dict(total=len(v),longer_than8=sum(x>8 for x in v),maximum=max(v,default=0),frames_in_longer_than8=sum(x for x in v if x>8)) for k,v in episode_lengths.items()},
        definitions=dict(retention='>=9 consecutive <.5 observations after burn-in, >=4 severe frames within that event, preceding 8 clear observations, following 4 clear observations, and CAD box fully inside image during the event',
            recovery='>=9 supervised consecutive <.3 observations, following 4 clear observations, and CAD box fully inside image during that severe event; may overlap retention',
            containment='Conservative CAD box-corner condition, not exact silhouette fraction'),all_eligible_windows=summary(windows),per_object={},manifests={})
    for obj in sorted(set(s['object_id'] for s in streams)):
        ws=[r for r in windows if r['object_id']==obj];report['per_object'][str(obj)]=dict(summary(ws),retention_physical_sequences=len({r['physical_sequence'] for r in ws if r['retention_ready']}),recovery_physical_sequences=len({r['physical_sequence'] for r in ws if r['recovery_ready']}))
    for item in a.manifest:
        name,path=item.split('=',1);data=Path(path).read_bytes();samples=json.loads(data);selected=[lookup[(r['stream'],r['start'])] for r in samples]
        report['manifests'][name]=dict(summary(selected),sha256=hashlib.sha256(data).hexdigest(),unique_streams=len({r['stream'] for r in samples}),unique_frame_windows=len({(r['stream'],r['start']) for r in samples}))
    (a.out/'windows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in windows));report['windows_sha256']=hashlib.sha256((a.out/'windows.jsonl').read_bytes()).hexdigest()
    (a.out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Natural occlusion coverage','',report['scope'],'',f"Full train: {m['streams']} streams, {m['frames']} frames, {len(values)} positions after first-frame exclusion.",
        f"Natural visibility <.5: {report['raw_natural_lt05_fraction']:.3%}; <.3: {report['raw_natural_lt03_fraction']:.3%}.",'',
        '| Training manifest | Windows | Natural <.5 targets | Natural <.3 targets | Long severe windows | Retention eligible | Recovery eligible |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in report['manifests'].items():lines.append(f"| {name} | {r['windows']} | {r['natural_lt05_fraction']:.3%} | {r['natural_lt03_fraction']:.3%} | {r['long_severe_windows']} | {r['retention_windows']} | {r['recovery_windows']} |")
    lines+=['',m['provenance_limit']];(a.out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
