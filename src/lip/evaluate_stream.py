"""Full-sequence streaming evaluation. GT is used only for initialization and metrics."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from lip.engine.config import check_data_gate
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import load_init,sha,source_hash
from lip.engine.stream_state import CACHE_CONTRACT
from lip.data.index import read_frame
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import center_pose,angle
from lip.evaluation.metrics import errors,summarize,visibility_bin
from lip.data.motion import motion_thresholds
from lip.evaluate import visibility


def merge_shards(root,world):
    rows=[];manifests=[]
    for rank in range(world):
        folder=root/f'rank{rank}';m=json.loads((folder/'manifest.json').read_text())
        if not m['completed']:raise RuntimeError('Incomplete shard')
        manifests.append(m);rows.extend(map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()))
    compare=('checkpoint_sha256','architecture_id','cache_contract','config','split_hash','mesh_hash','split','source_sha256')
    assert all(all(m[k]==manifests[0][k] for k in compare) for m in manifests)
    keys={(r['stream_id'],r['frame_index']) for r in rows};assert len(keys)==len(rows)
    rows.sort(key=lambda r:(r['stream_id'],r['frame_index']))
    save_reports(root,rows)
    m=manifests[0].copy();m.update(frames=len(rows),streams=sorted(s for m in manifests for s in m['streams']),shard_rank=None,shard_count=world)
    assert len(m['streams'])==len(set(m['streams']))
    assert m['streams']==m['expected_streams'] and len(rows)==m['expected_frames']
    m['population_verified']=True
    (root/'manifest.json').write_text(json.dumps(m,indent=2))


def save_reports(out,rows):
    report=summarize(rows)
    tracked=[r for r in rows if not r['initialization']]
    report['excluding_initialization']=summarize(tracked) if tracked else None
    hard=[r for r in tracked if r['moving'] and r['visibility'] is not None and r['visibility']<.3]
    report['moving_and_visibility_lt_03']=summarize(hard) if hard else dict(count=0)
    report['per_camera']={c:summarize([r for r in tracked if r['camera_id']==c]) for c in sorted(set(r['camera_id'] for r in tracked))}
    report['status_counts']={s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))}
    report['recovery']=dict(lost_to_not_lost=sum(bool(rows[i-1]['lost']) and not r['lost'] for i,r in enumerate(rows) if i and rows[i-1]['stream_id']==r['stream_id']),
        gt_resets_after_initialization=0,definition='lost streak >=5 ADD-S@.1d failures or explicit invalid state; recovery requires later valid successful outputs')
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--out',required=True)
    p.add_argument('--data-root',default=os.environ.get('DEX_YCB_DIR'));p.add_argument('--index-root',default='cache/dexycb_s0')
    p.add_argument('--split',choices=('val','test'),default='val');p.add_argument('--test-finalized',action='store_true')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--limit-streams',type=int);p.add_argument('--max-frames',type=int);p.add_argument('--merge',action='store_true');a=p.parse_args()
    out=Path(a.out)
    if a.merge:merge_shards(out,a.world_size);return
    if not a.config or not a.checkpoint or not a.data_root:p.error('config, checkpoint and data root required')
    if a.split=='test' and not a.test_finalized:p.error('Test is reserved for a frozen configuration; require --test-finalized')
    if not 0<=a.rank<a.world_size:p.error('Invalid shard')
    if (out/'predictions.jsonl').exists():raise FileExistsError('Refusing to overwrite evaluation')
    out.mkdir(parents=True,exist_ok=True);c=load_stream_config(a.config);audit=check_data_gate(a.index_root)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(c['cpu_threads'])
    model=make_model(c).to(device);ck=load_init(a.checkpoint,model,audit,c);model.eval();renderer=Renderer(device)
    streams=sorted([s for s in map(json.loads,(Path(a.index_root)/'streams.jsonl').read_text().splitlines()) if s['split']==a.split],key=lambda s:s['stream_id'])
    if a.limit_streams:streams=streams[:a.limit_streams]
    expected_streams=[s['stream_id'] for s in streams]
    expected_frames=sum(min(s['num_frames'],a.max_frames) if a.max_frames is not None else s['num_frames'] for s in streams)
    streams=streams[a.rank::a.world_size]
    manifest=dict(completed=False,architecture_id=c['architecture_id'],checkpoint_sha256=sha(a.checkpoint),
        checkpoint_stage_step=ck.get('new_stage_step',0),checkpoint_parent=ck.get('parent'),cache_contract=CACHE_CONTRACT,
        source_sha256=source_hash(),split=a.split,split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],config=c,
        initial_pose_source='GT first frame only',history_state_source='own committed prediction',fp_calls=0,critic_calls=0,
        depth_correction=False,full_sequences=a.max_frames is None,subset=a.limit_streams is not None,
        shard_rank=a.rank,shard_count=a.world_size,streams=[s['stream_id'] for s in streams],
        expected_streams=expected_streams,expected_frames=expected_frames,metric_pose_precision='fp32; SciPy nearest-neighbor search internal float64')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));rows=[]
    thresholds=motion_thresholds(a.index_root)
    with torch.no_grad(),(out/'predictions.jsonl').open('w') as writer:
        for s in streams:
            with np.load(Path(a.index_root)/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            with np.load(Path(a.index_root)/s['pose_cache']) as z:poses=z['poses'].copy();frames=z['frames'].copy();times=z['timestamps'].copy() if 'timestamps' in z else frames.astype('f8')/audit['fps']
            state=model.initialize(poses[0],mesh,s['intrinsics'],s['stream_id'],times[0],object_id=s['object_id'],camera_id=s['camera_serial'],mesh_hash=sha(Path(a.index_root)/s['mesh_cache']))
            centered=center_pose(torch.from_numpy(poses),torch.from_numpy(mesh['center']));streak=0
            for j,frame in enumerate(frames[:a.max_frames]):
                if j:
                    rgb,depth=read_frame(a.data_root,s,int(frame),audit['depth_scale_to_m'])
                    proposal,candidate=model.step(torch.from_numpy(rgb),torch.from_numpy(depth),times[j],state,renderer=renderer,
                        precision=c['precision'],image_size=c['image_size'],crop_expansion=c['crop_expansion'])
                    if proposal['status']=='ok':state=model.commit(proposal,candidate)
                    pred=proposal['pose_centered'];status=proposal['status'];needs=proposal['needs_reinit']
                else:pred=state.pose_centered;status='initialized';needs=False
                # Current GT does not enter the predictor or its state transaction.
                gt=centered[j];v=visibility(a.data_root,s,int(frame),gt.to(device),state.K,mesh,renderer)
                e=errors(pred.float().cpu(),gt.float(),mesh['vertices'],float(mesh['diameter']),dtype='f4')
                streak=streak+1 if not e['adds_01'] else 0;moving=False
                if j:
                    dt=times[j]-times[j-1]
                    moving=bool((gt[:3,3]-centered[j-1,:3,3]).norm()/float(mesh['diameter'])/dt>thresholds['center_d_per_sec'] or
                        angle(gt[:3,:3]@centered[j-1,:3,:3].T)/dt>thresholds['rotation_rad_per_sec'])
                row=dict(**e,stream_id=s['stream_id'],object_id=s['object_id'],camera_id=s['camera_serial'],frame_index=int(frame),
                    initialization=j==0,status=status,needs_reinit=needs,visibility=v,visibility_bin=visibility_bin(v),moving=moving,
                    lost=streak>=5 or needs,pose_centered=pred.cpu().tolist(),timestamp=float(times[j]),cache_bytes=state.cache.kv_bytes)
                writer.write(json.dumps(row)+'\n');writer.flush();rows.append(row)
            print(json.dumps(dict(completed_stream=s['stream_id'],frames=len(rows))),flush=True)
    save_reports(out,rows);manifest.update(completed=True,frames=len(rows));(out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
