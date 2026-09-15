"""Replay only the deterministic anchor admission policy on saved observations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('runtime','evaluation','events','index-root','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--arm');p.add_argument('--attention-probe',type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.runtime/'src'))
    from lip.models.stream_rk import AnchorBank
    from lip.engine.stream_state import FrameMeta
    torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False)
    m=json.loads((a.evaluation/'manifest.json').read_text());c=m['config'];assert m['completed'] and m['population_verified'] and m['frames']==23200
    assert c['keyframe_slots']==4 and c['memory_frames']==8 and m['fp_calls']==0
    event_report=json.loads(a.events.read_text());reference=event_report['reference'];raw=(a.evaluation/'predictions.jsonl').read_bytes()
    arm=a.arm or reference
    assert hashlib.sha256(raw).hexdigest()==event_report['prediction_sha256'][arm]
    events={(e['stream_id'],f):e for e in event_report['events'] for f in e['frames']}
    groups={}
    for r in map(json.loads,raw.splitlines()):groups.setdefault(r['stream_id'],[]).append(r)
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())};output=[];mismatches=[]
    with torch.no_grad():
        for sid,rs in sorted(groups.items()):
            rs.sort(key=lambda r:r['frame_index']);assert rs[0]['initialization'] and all(r['status']=='ok' for r in rs[1:])
            s=streams[sid];d=s['mesh_diameter']
            with np.load(a.index_root/s['mesh_cache']) as z:center=z['center'].copy()
            with np.load(a.index_root/s['pose_cache']) as z:gt={int(f):t.copy() for f,t in zip(z['frames'],z['poses'])}
            for pose in gt.values():pose[:3,3]+=pose[:3,:3]@center
            lookup={r['frame_index']:r for r in rs};bank=AnchorBank.empty(torch.zeros(1,256),c['keyframe_slots']);prior_errors={};previous=rs[0]
            for r in rs[1:]:
                frame=r['frame_index'];meta=FrameMeta(torch.tensor([r['timestamp']],dtype=torch.float64),torch.tensor([frame]),torch.zeros(1,dtype=torch.long),torch.ones(1,17,dtype=torch.bool),torch.zeros(1,17))
                age=frame-bank.frame_id;valid=bank.valid&(age>=c['memory_frames'])&(age<=c['keyframe_max_age'])&(bank.timestamp<=meta.timestamp[:,None])
                read=int(valid.sum())
                if read!=r['anchors_read']:mismatches.append(dict(stream_id=sid,frame=frame,actual=read,saved=r['anchors_read']))
                e=events.get((sid,frame));anchors=[]
                for slot in torch.nonzero(valid[0],as_tuple=False).flatten().tolist():
                    source=int(bank.frame_id[0,slot]);sr=lookup[source];visible=sr['visibility'] is not None and sr['visibility']>=.5
                    anchors.append(dict(frame=source,age=frame-source,quality=float(bank.quality[0,slot]),visibility=sr['visibility'],
                        source_posterior_adds005=bool(sr['adds_005']),prior_center_mm=prior_errors[source],
                        clear_pre_episode=bool(e and source<e['start'] and visible),
                        successful_clear_pre_episode=bool(e and source<e['start'] and visible and sr['adds_005'])))
                output.append(dict(stream_id=sid,frame=frame,object_id=r['object_id'],visibility=r['visibility'],adds_005=bool(r['adds_005']),
                    reference_entry=e['reference_entry'] if e else None,episode_start=e['start'] if e else None,anchors_read=read,anchors=anchors,
                    clear_pre_episode_anchors=sum(x['clear_pre_episode'] for x in anchors),successful_clear_pre_episode_anchors=sum(x['successful_clear_pre_episode'] for x in anchors)))
                base=torch.tensor(previous['pose_centered'],dtype=torch.float32);prior_errors[frame]=float(np.linalg.norm(base[:3,3].numpy()-gt[frame][:3,3])*1000)
                features=dict(T_base_centered=base[None],state_input=torch.zeros(1,24),source_xy=torch.zeros(1,16,2))
                bank=bank.insert(torch.zeros(1,256),torch.tensor([r['observation_support']],dtype=torch.float32),meta,features,c['keyframe_min_gap'],c['keyframe_max_age']);previous=r
    assert len(output)==22880 and not mismatches,mismatches[:5]
    identity_check=None
    if a.attention_probe:
        probe=json.loads((a.attention_probe/'attention.json').read_text());assert probe['passed'] and probe['checkpoint_sha256']==m['checkpoint_sha256']
        actual=[json.loads(x) for x in (a.attention_probe/'frames.jsonl').read_text().splitlines()]
        reconstructed={(r['stream_id'],r['frame']):r for r in output}
        assert all([x['frame'] for x in reconstructed[(r['stream_id'],r['frame'])]['anchors']]==r['anchor_frames'] for r in actual)
        identity_check=dict(passed=True,frames=len(actual),streams=len(probe['streams']),scope='Exact selected frame IDs versus directly instrumented model cache, including slot order')
    low=[r for r in output if r['episode_start'] is not None];stable=[r for r in low if r['reference_entry']=='stable_good'];failed=[r for r in stable if not r['adds_005']]
    summary={}
    for name,rows in [('all',output),('low_visibility',low),('stable_good_entry',stable),('stable_good_entry_failed_frames',failed)]:
        summary[name]=dict(frames=len(rows),zero_usable_anchors=sum(r['anchors_read']==0 for r in rows),
            with_clear_pre_episode_anchor=sum(r['clear_pre_episode_anchors']>0 for r in rows),
            with_successful_clear_pre_episode_anchor=sum(r['successful_clear_pre_episode_anchors']>0 for r in rows))
    report=dict(completed=True,scope='CPU replay of admission indices only, not a network replay or changed tracker. Exact saved scalar quality/timestamps and actual AnchorBank implementation; dummy latents/crop data cannot influence selection. All 22880 saved read counts agree. GT visibility and pose metrics label sources only after selection.',
        checkpoint_sha256=m['checkpoint_sha256'],prediction_sha256=hashlib.sha256(raw).hexdigest(),
        selection_source_sha256=hashlib.sha256((a.runtime/'src/lip/models/stream_rk.py').read_bytes()).hexdigest(),read_count_mismatches=mismatches,
        arm=arm,direct_anchor_identity_verification=identity_check,populations=summary)
    (a.out/'retention.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (a.out/'frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in output));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
