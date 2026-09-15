"""Native FP tracking from the exact non-GT initializers shared with LIP."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.config import check_data_gate
from lip.engine.stream_checkpoint import sha,source_hash
from lip.geometry.so3 import center_pose
from lip.integrations.foundationpose import FoundationPoseAdapter
from streaming_bop_utils import legal_pose
from val_non_gt_common import val_streams,validate_initializers,NativeSplitInferenceGuard


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('fp-root','initializers','data-root','index-root','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-weight-sha',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--limit-streams',type=int);p.add_argument('--max-frames',type=int);p.add_argument('--stream-ids',type=Path);a=p.parse_args()
    audit=check_data_gate(a.index_root);streams=val_streams(a.index_root);initial=validate_initializers(json.loads(a.initializers.read_text()),streams,audit)
    assert not initial.get('subset') and len(streams)==320 and sum(s['num_frames'] for s in streams)==23200
    assert 0<=a.rank<a.world
    if a.stream_ids:
        ids=json.loads(a.stream_ids.read_text());assert len(ids)==len(set(ids)) and set(ids)<={s['stream_id'] for s in streams}
        streams=[s for s in streams if s['stream_id'] in ids]
    if a.limit_streams:streams=streams[:a.limit_streams]
    population=[s['stream_id'] for s in streams];expected=sum(min(s['num_frames'],a.max_frames) if a.max_frames else s['num_frames'] for s in streams)
    streams=streams[a.rank::a.world];a.out.mkdir(parents=True,exist_ok=False)
    guard=NativeSplitInferenceGuard(a.data_root,a.index_root,a.fp_root,streams,'val',allow_foundationpose=True)
    sys.addaudithook(guard);sys.dont_write_bytecode=True;sys.path.insert(0,str(a.fp_root))
    torch.set_num_threads(2);cv2.setNumThreads(0);torch.manual_seed(42);np.random.seed(42)
    weight=a.fp_root/'weights/2023-10-28-18-33-37/model_best.pth';assert sha(weight)==a.expected_weight_sha
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    import nvdiffrast.torch as dr
    class TrackingOnly(FoundationPose):
        def make_rotation_grid(self,*args,**kwargs):pass
    logging.getLogger().setLevel(logging.WARNING)
    refiner=PoseRefinePredictor();refiner.model.eval().requires_grad_(False);context=dr.RasterizeCudaContext();pool={}
    config=dict(method='FoundationPose track_one',iterations=2,precision='official FP16 autocast',initial_refinement=False,registration=False,object_setup_seed=42)
    (a.out/'config.json').write_text(json.dumps(config,indent=2));h=hashlib.sha256()
    for path in sorted(a.fp_root.rglob('*.py')):
        if not any(v in path.parts for v in ('weights','__pycache__','.git')):h.update(str(path.relative_to(a.fp_root)).encode());h.update(path.read_bytes())
    receipt=dict(completed=False,split='val',architecture_id='foundationpose_tracking_v1',source_sha256=source_hash(),
        checkpoint_sha256=sha(weight),config=config,config_sha256=sha(a.out/'config.json'),initializers_sha256=sha(a.initializers),
        initializer_backend=initial['backend'],initializer_checkpoint_sha256=initial['checkpoint_sha256'],
        split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],initialization_uses_gt_pose=False,
        expected_streams=population,expected_frames=expected,streams=[s['stream_id'] for s in streams],rank=a.rank,world=a.world,
        frames=0,initialized_streams=0,missing_pose_frames=0,fp_calls=0,fp_initialization_calls=0,fp_iterations=2,
        gt_pose_reads=0,gt_mask_reads=0,hand_annotation_reads=0,gt_resets=0,held_updates=0,
        subset=bool(a.limit_streams or a.max_frames or a.stream_ids),fp_source_sha256=h.hexdigest(),entrypoint_sha256=sha(Path(__file__)),
        foundationpose_commit=subprocess.check_output(['git','-C',str(a.fp_root),'rev-parse','HEAD'],text=True).strip(),
        protocol='Keep the same first legal PoseCNN pose as LIP, then official track_one(iteration=2) on every following RGB-D frame with its own state. No registration, GT reset, or redetection. Missing initialization remains in the denominator.',
        fp_mesh_diameters={},timing_scope='Concurrent full-val throughput; not an isolated latency benchmark.')
    start=time.time()
    def save():
        receipt.update(seconds=time.time()-start,access_audit=guard.snapshot())
        temp=a.out/'manifest.tmp';temp.write_text(json.dumps(receipt,indent=2));temp.replace(a.out/'manifest.json')
    with torch.no_grad(),(a.out/'predictions.jsonl').open('w') as writer:
        for s in streams:
            sid=s['stream_id'];external=initial['initializers'][sid];K=np.asarray(s['intrinsics'],dtype='f4');prior=None
            with np.load(a.index_root/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            count=min(s['num_frames'],a.max_frames) if a.max_frames else s['num_frames']
            if external is not None and external['frame_index']<count:
                oid=s['object_id']
                if oid not in pool:
                    raw=trimesh.load(a.data_root/s['mesh_path'],process=False,force='mesh');raw.vertices*=audit['mesh_scale_to_m']
                    assert np.allclose((raw.vertices.max(0)+raw.vertices.min(0))/2,mesh['center'],atol=1e-6)
                    rng=np.random.get_state();np.random.seed(42)
                    try:est=TrackingOnly(raw.vertices,raw.vertex_normals,mesh=raw,scorer=object(),refiner=refiner,glctx=context,debug=0,debug_dir=str(a.out/'fp_debug'))
                    finally:np.random.set_state(rng)
                    est.diameter=float(est.diameter);pool[oid]=est;receipt['fp_mesh_diameters'][str(oid)]=est.diameter
                est=pool[oid];adapter=FoundationPoseAdapter(est,'cuda')
            for frame in range(count):
                is_initial=external is not None and frame==external['frame_index']
                row=dict(stream_id=sid,frame_index=frame,object_id=s['object_id'],camera_id=s['camera_serial'],timestamp=frame/audit['fps'],
                    initialization=is_initial,pose_original=None,pose_centered=None,status='uninitialized',needs_reinit=True)
                if external is not None and frame>=external['frame_index']:
                    color=a.data_root/s['relative_dir']/f'color_{frame:06d}.jpg';dep=a.data_root/s['relative_dir']/f'aligned_depth_to_color_{frame:06d}.png'
                    guard.check(color,native=True);guard.check(dep,native=True);rgb=cv2.imread(str(color));depth=cv2.imread(str(dep),-1)
                    assert rgb is not None and depth is not None
                    rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB);depth=depth.astype('f4')*audit['depth_scale_to_m']
                    if is_initial:
                        prior=np.asarray(external['pose_original'],dtype='f4');adapter.accept(prior);status='initialized_posecnn:ok';receipt['initialized_streams']+=1
                    else:
                        result=np.asarray(est.track_one(rgb=rgb,depth=depth,K=K,iteration=2),dtype='f4');receipt['fp_calls']+=1
                        if legal_pose(result):prior=result;status='ok'
                        else:adapter.accept(prior);status='invalid_fp_pose_held';receipt['held_updates']+=1
                    centered=center_pose(torch.from_numpy(prior.copy()),torch.from_numpy(mesh['center'].copy()).float())
                    row.update(pose_original=prior.tolist(),pose_centered=centered.tolist(),status=status,needs_reinit=status not in ('ok','initialized_posecnn:ok'))
                else:receipt['missing_pose_frames']+=1
                writer.write(json.dumps(row,allow_nan=False)+'\n');receipt['frames']+=1
            writer.flush();save();print(json.dumps(dict(stream=sid,frames=receipt['frames'],fp_calls=receipt['fp_calls'])),flush=True)
    assert not any(v for k,v in guard.snapshot()['counts'].items() if k.startswith('denied_'))
    receipt.update(completed=True,predictions_sha256=sha(a.out/'predictions.jsonl'));save()


if __name__=='__main__':main()
