"""Independent native-val inference; no GT access, with sealed prediction shards."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.engine.jepa_checkpoint import load_core,sha
from lip.engine.object_jepa_checkpoint import atomic_json
from lip.unified.runtime import initialize,step
from lip.engine.config import check_data_gate
from lip.engine.stream_checkpoint import source_hash
from lip.jepa.config import config_hash
from lip.unified.renderer import FullTextureRenderer as AppearanceRenderer
from lip.geometry.so3 import original_pose
from val_non_gt_common import val_streams,validate_initializers,NativeValGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8)
    p.add_argument('--disable-history',action='store_true');p.add_argument('--interface-smoke',action='store_true')
    p.add_argument('--limit-streams',type=int);p.add_argument('--max-frames',type=int);a=p.parse_args()
    if a.interface_smoke!=bool(a.limit_streams and a.max_frames):raise ValueError('Subset bounds require explicit interface smoke')
    c=yaml.safe_load(a.config.read_text());index=Path(c['paths']['index_root']);root=Path(c['paths']['data_root']);audit=check_data_gate(index)
    if c['runtime'].get('deterministic_algorithms',False):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory=c['runtime'].get('fill_uninitialized_memory',True)
    all_streams=val_streams(index);assert len(all_streams)==320 and sum(s['num_frames'] for s in all_streams)==23200
    init_path=Path(c['paths']['val_initializers']);initials=validate_initializers(json.loads(init_path.read_text()),all_streams,audit)
    selected=all_streams[:a.limit_streams] if a.interface_smoke else all_streams;streams=selected[a.rank::a.world]
    torch.cuda.set_device(0 if torch.cuda.device_count()==1 else a.rank);torch.set_num_threads(2);torch.manual_seed(c['seed'])
    fp_encoder=c['architecture_id']=='stream_dino_fp_staticutonia_jepa_rgbd_v3'
    fp_root=Path(c['paths'].get('fp_root','/mnt/why/dexycb_lip/third_party/FoundationPose'))
    fp_files=[fp_root/'learning/models/network_modules.py',fp_root/'learning/models/refine_network.py',
              Path(c['paths']['fp_checkpoint']),Path(c['paths']['fp_checkpoint']).with_name('config.yml')] if fp_encoder else []
    guard=NativeValGuard(root,index,fp_root,streams,fp_encoder_files=fp_files);sys.addaudithook(guard);sys.dont_write_bytecode=True
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['config_hash']!=config_hash(c):raise ValueError('Checkpoint/config identity mismatch')
    if checkpoint['p0_only'] and not a.interface_smoke:raise ValueError('P0 cannot become formal full-val candidate')
    for key in ['split_hash','mesh_hash']:
        if checkpoint['provenance'][key]!=audit[key]:raise ValueError('Evaluation data identity mismatch')
    model=build_model(c);load_core(model,checkpoint['model']);model.weights_version=sha(a.checkpoint)
    model.eval();renderer=AppearanceRenderer('cuda');store=make_store(c,model)
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    manifest=dict(completed=False,split='val',architecture_id=model.architecture_id,source_sha256=source_hash(),
        checkpoint_sha256=sha(a.checkpoint),config=c,config_sha256=config_hash(c),seed=c['seed'],stage_optimizer_steps=checkpoint['step'],
        initializers_sha256=sha(init_path),initializer_backend=initials['backend'],initializer_checkpoint_sha256=initials['checkpoint_sha256'],
        split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],initialization_uses_gt_pose=False,
        expected_streams=sorted(s['stream_id'] for s in selected),expected_frames=sum(min(s['num_frames'],a.max_frames) if a.interface_smoke else s['num_frames'] for s in selected),
        streams=[s['stream_id'] for s in streams],rank=a.rank,world=a.world,frames=0,initialized_streams=0,missing_pose_frames=0,
        fp_calls=0,fp_encoder_image_pairs=0,gt_pose_reads=0,gt_mask_reads=0,hand_annotation_reads=0,gt_resets=0,subset=a.interface_smoke,
        interface_smoke=a.interface_smoke,jepa_enabled=True,history_enabled=not a.disable_history,rendered_dino_images=0,current_dino_images=0,rejected_updates=0,max_memory_tokens=0,
        max_history_frames=0,protocol='Native PoseCNN; own causal pose feedback; no later GT reset; '+model.architecture_id+'; full texture; no teacher',
        entrypoint_sha256=sha(__file__))
    def record():
        manifest.update(seconds=time.monotonic()-started,access_audit=guard.snapshot());atomic_json(a.out/'manifest.json',manifest)
    with torch.no_grad(),(a.out/'predictions.jsonl').open('w') as writer:
        for s in streams:
            sid=s['stream_id'];init=initials['initializers'][sid];state=None
            with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            mesh['vertices']=torch.tensor(mesh['vertices'],device='cuda');mesh['faces']=torch.tensor(mesh['faces'],device='cuda',dtype=torch.int32)
            cad=store.get(root/s['mesh_path'],mesh)
            k=torch.tensor(s['intrinsics'],device='cuda');center=torch.tensor(mesh['center'],device='cuda')
            count=min(s['num_frames'],a.max_frames) if a.interface_smoke else s['num_frames']
            for frame in range(count):
                is_initial=init is not None and frame==init['frame_index'];timestamp=frame/audit['fps']
                row=dict(stream_id=sid,frame_index=frame,object_id=s['object_id'],camera_id=s['camera_serial'],timestamp=timestamp,
                    initialization=is_initial,pose_original=None,pose_centered=None,status='uninitialized',needs_reinit=True,measurement_valid=False)
                if init is not None and frame>=init['frame_index']:
                    rgb_path=root/s['relative_dir']/f'color_{frame:06d}.jpg';depth_path=root/s['relative_dir']/f'aligned_depth_to_color_{frame:06d}.png'
                    guard.check(rgb_path,native=True);guard.check(depth_path,native=True)
                    im=cv2.imread(str(rgb_path));d=cv2.imread(str(depth_path),-1)
                    if im is None or d is None:raise FileNotFoundError(rgb_path)
                    rgb=torch.tensor(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy(),device='cuda').float()/255
                    depth=torch.tensor(d.astype('f4')[None],device='cuda')*audit['depth_scale_to_m'];start=time.monotonic()
                    if is_initial:
                        state=initialize(model,rgb,depth,init['pose_original'],mesh,k,timestamp,sid,renderer,cad,enabled=not a.disable_history)
                        status='initialized_posecnn';valid=False;manifest['initialized_streams']+=1
                    else:
                        result,state=step(model,rgb,depth,timestamp,state,renderer);status=result['status'];valid=status=='ok'
                        manifest['rejected_updates']+=int(not valid)
                    torch.cuda.synchronize();pose=state.pose_centered
                    row.update(pose_centered=pose.cpu().tolist(),pose_original=init['pose_original'] if is_initial else original_pose(pose,center).cpu().tolist(),
                        status=status,needs_reinit=False,measurement_valid=valid,measured_step_ms=(time.monotonic()-start)*1000)
                    manifest['current_dino_images']+=1;manifest['rendered_dino_images']+=1
                    manifest['fp_encoder_image_pairs']+=int(fp_encoder)
                    manifest['max_history_frames']=max(manifest['max_history_frames'],len(state.memory.contexts))
                    manifest['max_memory_tokens']=max(manifest['max_memory_tokens'],sum(r.features.shape[1] for r in state.memory.objects+state.memory.contexts))
                else:manifest['missing_pose_frames']+=1
                writer.write(json.dumps(row,allow_nan=False)+'\n');manifest['frames']+=1
            writer.flush();record()
    if guard.snapshot()['counts']['validated_native_image_reads']!=2*(manifest['frames']-manifest['missing_pose_frames']):raise AssertionError('Image-read counter mismatch')
    if any(v for k,v in guard.snapshot()['counts'].items() if k.startswith('denied_')):raise AssertionError('Inference access boundary violation')
    manifest.update(completed=True,predictions_sha256=sha(a.out/'predictions.jsonl'));record()


if __name__=='__main__':main()
