"""Partition occlusion failures by prior observation availability, posthoc only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from diagnose_occlusion_events import extract_episodes,successful,visibility_class


def classify_entry(event, rows):
    ordered=sorted(rows,key=lambda r:r['frame_index']);lookup={r['frame_index']:r for r in ordered}
    pre=[lookup[f] for f in event['pre_frames']]
    if event['pre_complete']:
        category='stable_good' if all(successful(r) for r in pre) else 'clear_but_inaccurate'
    else:
        run=0;ever_three=False;previous=None
        for row in ordered:
            if row['frame_index']>=event['start']:break
            clear=not row['initialization'] and visibility_class(row)=='clear'
            run=(run+1 if previous is not None and row['frame_index']==previous+1 else 1) if clear else 0
            ever_three|=run>=3;previous=row['frame_index']
        category='interrupted_clear_context' if ever_three else 'no_prior_three_clear_observations'
    init=ordered[0]
    if not init['initialization']:raise ValueError('Initialization record required')
    positions={row['frame_index']:i for i,row in enumerate(ordered)}
    return dict(category=category,start_position=positions[event['start']],starts_within_first8=positions[event['start']]<=8,
        initial_image_clear=visibility_class(init)=='clear',initial_pose_adds005=bool(init['adds_005']))


def summarize(rows):
    objects=defaultdict(list)
    for row in rows:objects[row['object_id']].append(row)
    return dict(frames=len(rows),objects=len(objects),failures=sum(not successful(r) for r in rows),
        micro_success=sum(successful(r) for r in rows)/len(rows) if rows else None,
        object_macro_success=sum(sum(successful(r) for r in rs)/len(rs) for rs in objects.values())/len(objects) if objects else None)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--reference',required=True,type=Path);p.add_argument('--evaluation',action='append',default=[],help='name=folder')
    p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    folders={'reference':a.reference,**{n:Path(path) for n,path in (arg.split('=',1) for arg in a.evaluation)}}
    data={};manifests={};hashes={}
    for name,folder in folders.items():
        m=json.loads((folder/'manifest.json').read_text());assert m['completed'] and m['population_verified'] and m['frames']==23200 and m['split']=='val' and m['fp_calls']==0
        raw=(folder/'predictions.jsonl').read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest();manifests[name]=m
        data[name]={(r['stream_id'],r['frame_index']):r for r in map(json.loads,raw.splitlines())};assert len(data[name])==23200
    reference=data['reference'];groups=defaultdict(list)
    for row in reference.values():groups[row['stream_id']].append(row)
    for name,rows in data.items():
        assert set(rows)==set(reference)
        assert all(rows[k]['visibility']==reference[k]['visibility'] for k in reference)
        assert all(manifests[name][k]==manifests['reference'][k] for k in ('split_hash','mesh_hash','initial_poses_sha256'))
    events=extract_episodes(list(reference.values()));event_of={}
    for event in events:
        event.update(classify_entry(event,groups[event['stream_id']]))
        for frame in event['frames']:event_of[(event['stream_id'],frame)]=event
    populations={'all_tracked':[key for key,row in reference.items() if not row['initialization']],
        'visibility_lt05':list(event_of),'visibility_lt03':[key for key in event_of if reference[key]['visibility']<.3]}
    for category in sorted({e['category'] for e in events}):
        keys=[key for key,event in event_of.items() if event['category']==category]
        populations[category]=keys
        populations[category+'_severe']=[key for key in keys if reference[key]['visibility']<.3]
    populations['no_three_clear_but_initial_image_clear']=[key for key,e in event_of.items() if e['category']=='no_prior_three_clear_observations' and e['initial_image_clear']]
    populations['no_three_clear_initial_clear_and_pose_accurate']=[key for key,e in event_of.items() if e['category']=='no_prior_three_clear_observations' and e['initial_image_clear'] and e['initial_pose_adds005']]
    populations['early_occlusion']=[key for key,e in event_of.items() if e['starts_within_first8']]
    positions={}
    for sid,rows in groups.items():
        for i,row in enumerate(sorted(rows,key=lambda r:r['frame_index'])):positions[(sid,row['frame_index'])]=i
    for suffix,condition in [('first8',lambda i:1<=i<=8),('after8',lambda i:i>8)]:
        for name in ('all_tracked','visibility_lt05','visibility_lt03'):
            populations[name+'_'+suffix]=[key for key in populations[name] if condition(positions[key])]
    report=dict(completed=True,scope='Posthoc decomposition of the same controlled s0 val, zero FP. Entry categories use the frozen reference only. Initialization-image availability is not proof that priming would improve tracking.',
        definition='Stable entry: three immediately prior clear and accurate non-initialization frames. No-prior-three-clear counts the entire past prefix, excluding the initialization image. Successful means ADD-S<.05d and status ok.',
        prediction_sha256=hashes,checkpoints={k:v['checkpoint_sha256'] for k,v in manifests.items()},
        groups={name:{arm:summarize([rows[key] for key in keys]) for arm,rows in data.items()} for name,keys in populations.items()},events=events)
    (a.out/'origins.json').write_text(json.dumps(report,indent=2));lines=['# Occlusion failure origins','',report['scope'],'',
        '| Fixed reference population | Frames | '+ ' | '.join(f'{name} strict success (%)' for name in data)+' |','|---|---:|'+'---:|'*len(data)]
    for name,values in report['groups'].items():
        if not values['reference']['frames']:continue
        lines.append(f"| {name} | {values['reference']['frames']} | "+' | '.join(f"{100*r['object_macro_success']:.3f}" for r in values.values())+' |')
    (a.out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
