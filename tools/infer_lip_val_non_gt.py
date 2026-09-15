"""Native full s0 val with real external initializers and no inference GT reads."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.config import check_data_gate
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import load_init,sha,source_hash
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import original_pose
from standalone_bop_state import prime_initial_observation,cache_summary
from streaming_bop_utils import hold_failed_step
from val_non_gt_common import val_streams,NativeValGuard,validate_initializers


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('config','checkpoint','initializers','data-root','index-root','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--startup-iterations',type=int,choices=(1,2),default=1);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--limit-streams',type=int);p.add_argument('--max-frames',type=int)
    p.add_argument('--stream-ids',type=Path,help='Explicit val-only subset for interface checks')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=p.parse_args()
    assert 0<=a.rank<a.world
    streams=val_streams(a.index_root);audit=check_data_gate(a.index_root)
    assert len(streams)==320 and sum(s['num_frames'] for s in streams)==23200
    initial=validate_initializers(json.loads(a.initializers.read_text()),streams,audit)
    if initial.get('subset'):raise ValueError('Subset initializer output cannot define full val')
    if a.stream_ids:
        ids=json.loads(a.stream_ids.read_text());assert len(ids)==len(set(ids)) and set(ids)<={s['stream_id'] for s in streams}
        streams=[s for s in streams if s['stream_id'] in ids]
    if a.limit_streams:streams=streams[:a.limit_streams]
    population=[s['stream_id'] for s in streams];expected=sum(min(s['num_frames'],a.max_frames) if a.max_frames else s['num_frames'] for s in streams)
    streams=streams[a.rank::a.world];a.out.mkdir(parents=True,exist_ok=False)
    guard=NativeValGuard(a.data_root,a.index_root,'/mnt/why/dexycb_lip/third_party/FoundationPose',streams);sys.addaudithook(guard)
    c=load_stream_config(a.config);device=torch.device(a.device)
    if a.device=='cpu' and c['precision']!='fp32':raise ValueError('CPU requires explicit FP32 config')
    torch.set_num_threads(2);cv2.setNumThreads(0);torch.manual_seed(42)
    model=make_model(c).to(device).eval();load_init(a.checkpoint,model,audit,c);renderer=Renderer(device)
    receipt=dict(startup_iterations=a.startup_iterations,extra_inner_attempts=0,inner_fallbacks=0,completed=False,split='val',architecture_id=c['architecture_id'],source_sha256=source_hash(),
        checkpoint_sha256=sha(a.checkpoint),config=c,config_sha256=sha(a.config),initializers_sha256=sha(a.initializers),
        initializer_backend=initial['backend'],initializer_checkpoint_sha256=initial['checkpoint_sha256'],
        split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],initialization_uses_gt_pose=False,
        expected_streams=population,expected_frames=expected,streams=[s['stream_id'] for s in streams],
        rank=a.rank,world=a.world,frames=0,initialized_streams=0,missing_pose_frames=0,fp_calls=0,
        gt_pose_reads=0,gt_mask_reads=0,hand_annotation_reads=0,gt_resets=0,
        subset=bool(a.limit_streams or a.max_frames or a.stream_ids),max_history_frames=0,max_valid_anchors=0,
        protocol='First legal real PoseCNN initializer; encode initial RGB-D and retain external pose, then full own-state causal tracking. Missing initialization produces no pose and remains in the denominator. No re-detection or GT reset. Invalid proposals hold pose and advance clock.')
    started=time.time()
    def save():
        receipt.update(seconds=time.time()-started,access_audit=guard.snapshot())
        tmp=a.out/'manifest.tmp';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(a.out/'manifest.json')
    def step(state,rgb,depth,timestamp):
        from lip.engine.intraframe import step_iterated
        iterations=a.startup_iterations if 1<=state.frame_id<=8 else 1
        proposal,next_state=step_iterated(model,torch.from_numpy(rgb.transpose(2,0,1).copy()),torch.from_numpy(depth[None]),timestamp,state,iterations=iterations,
            renderer=renderer,precision=c['precision'],image_size=c['image_size'],crop_expansion=c['crop_expansion'])
        receipt['extra_inner_attempts']+=proposal['inner_iterations_attempted']-1
        receipt['inner_fallbacks']+=int(proposal['inner_failure'] is not None)
        if proposal['status']=='ok':state=model.commit(proposal,next_state)
        else:state=hold_failed_step(state,timestamp)
        return state,proposal.get('pose_original'),proposal['status'],proposal
    with torch.no_grad(),(a.out/'predictions.jsonl').open('w') as writer:
        for stream in streams:
            sid=stream['stream_id'];external=initial['initializers'][sid];state=None;K=np.asarray(stream['intrinsics'],dtype='f4')
            with np.load(a.index_root/stream['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            count=min(stream['num_frames'],a.max_frames) if a.max_frames else stream['num_frames']
            for frame in range(count):
                timestamp=frame/audit['fps'];is_initial=external is not None and frame==external['frame_index']
                row=dict(stream_id=sid,frame_index=frame,object_id=stream['object_id'],camera_id=stream['camera_serial'],
                    initialization=is_initial,timestamp=timestamp,pose_original=None,pose_centered=None,status='uninitialized',needs_reinit=True)
                proposal={}
                if external is not None and frame>=external['frame_index']:
                    color=a.data_root/stream['relative_dir']/f'color_{frame:06d}.jpg';depthpath=a.data_root/stream['relative_dir']/f'aligned_depth_to_color_{frame:06d}.png'
                    guard.check(color,native=True);guard.check(depthpath,native=True)
                    image=cv2.imread(str(color));dep=cv2.imread(str(depthpath),-1)
                    assert image is not None and dep is not None
                    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);depth=dep.astype('f4')*audit['depth_scale_to_m']
                    if is_initial:
                        def observe(current):
                            new,pose,status,_=step(current,rgb,depth,timestamp);return new,pose,status,0
                        state,warm_status=prime_initial_observation(model,np.asarray(external['pose_original'],dtype='f4'),mesh,K,sid,
                            timestamp,audit['fps'],depth.shape,stream['object_id'],stream['camera_serial'],observe)
                        status='initialized_posecnn:'+warm_status;receipt['initialized_streams']+=1
                    else:state,_,status,proposal=step(state,rgb,depth,timestamp)
                    row.update(pose_original=original_pose(state.pose_centered,state.mesh['center'].float()).cpu().tolist(),
                        pose_centered=state.pose_centered.cpu().tolist(),status=status,needs_reinit=status not in ('ok',) and not is_initial)
                    for key in ('inner_iterations_requested','inner_iterations_attempted','inner_iterations_accepted','inner_failure'):
                        if key in proposal:row[key]=proposal[key]
                    if is_initial:row['pose_original']=np.asarray(external['pose_original'],dtype='f4').tolist()
                    if 'rotation_alignment_rotvec' in proposal:row['rotation_alignment_angle_deg']=float(proposal['rotation_alignment_rotvec'].float().norm()*180/torch.pi)
                    summary=cache_summary(state.cache);row['cache']=summary
                    receipt['max_history_frames']=max(receipt['max_history_frames'],summary['recent_frames'])
                    receipt['max_valid_anchors']=max(receipt['max_valid_anchors'],summary['valid_anchors'])
                    for key in ('rotation_anchor_coefficient','rotation_anchor_fraction','rotation_anchor_gap_norm',
                                'reference_write_rotation_coefficient','reference_write_center_coefficient','observation_support'):
                        if key in proposal:row[key]=float(proposal[key])
                else:receipt['missing_pose_frames']+=1
                writer.write(json.dumps(row,allow_nan=False)+'\n');receipt['frames']+=1
            writer.flush();save();print(json.dumps(dict(stream_id=sid,frames=receipt['frames'])),flush=True)
    assert not any(v for k,v in guard.snapshot()['counts'].items() if k.startswith('denied_'))
    receipt.update(completed=True,predictions_sha256=sha(a.out/'predictions.jsonl'));save()


if __name__=='__main__':main()
