"""Released PoseCNN RGB s0 weights on native val; never read pose/mask labels."""
import argparse
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types
import cv2
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.config import check_data_gate
from lip.engine.stream_checkpoint import sha
from lip.data.index import CLASSES
from val_non_gt_common import NativeValGuard,val_streams,first_legal_for_object


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('source','native-build','deps','checkpoint','data-root','index-root','out'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--expected-checkpoint-sha',required=True);p.add_argument('--limit-streams',type=int)
    p.add_argument('--max-scan-frames',type=int);a=p.parse_args();root=Path(__file__).resolve().parents[1]
    assert sha(a.checkpoint)==a.expected_checkpoint_sha
    streams=val_streams(a.index_root);audit=check_data_gate(a.index_root)
    assert len(streams)==320 and sum(s['num_frames'] for s in streams)==23200
    if a.limit_streams:streams=streams[:a.limit_streams]
    a.out.mkdir(parents=True,exist_ok=False)
    guard=NativeValGuard(a.data_root,a.index_root,'/mnt/why/dexycb_lip/third_party/FoundationPose',streams)
    sys.addaudithook(guard);torch.set_num_threads(2);cv2.setNumThreads(0)
    torch.manual_seed(3);np.random.seed(3);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path[:0]=[str(a.native_build),str(a.deps),str(a.source/'lib')]
    native=importlib.import_module('lip_posecnn_inference_cuda');sys.modules['posecnn_cuda']=native
    build=json.loads((a.native_build/'build_receipt.json').read_text());assert sha(native.__file__)==build['module_sha256']
    # The official eval forward calls auxiliary target helpers even though their
    # outputs are unused. Empty placeholders select their no-GT branch. If an
    # overlap computation is ever reached, abort rather than invent supervision.
    overlap=types.ModuleType('utils.cython_bbox')
    def forbidden_overlap(*args,**kwargs):raise RuntimeError('GT overlap is unavailable in non-GT initialization')
    overlap.bbox_overlaps=forbidden_overlap;sys.modules['utils.cython_bbox']=overlap
    from easydict import EasyDict
    from fcn.config import cfg,_merge_a_into_b
    specification=yaml.load((a.source/'experiments/cfgs/dex_ycb.yml').read_text(),Loader=yaml.FullLoader)
    _merge_a_into_b(EasyDict(specification),cfg);cfg.MODE='TEST';cfg.TEST.POSE_REFINE=False;cfg.TRAIN.GPUNUM=1
    from networks.PoseCNN import PoseCNN
    from utils.nms import nms
    from utils.se3 import allocentric2egocentric
    from transforms3d.quaternions import quat2mat
    ids=list(cfg.TRAIN.CLASSES);assert ids==[0,*range(1,19),20,21]
    with (a.out/'model_load.log').open('w') as log,contextlib.redirect_stdout(log):model=PoseCNN(len(ids),cfg.TRAIN.NUM_UNITS)
    weights=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(weights,strict=True);model=model.cuda().eval()
    assert all(torch.isfinite(t).all() for t in model.state_dict().values())
    extents=np.zeros((len(ids),3),dtype='f4')
    for i,oid in enumerate(ids[1:],1):
        points=np.loadtxt(a.data_root/'models'/CLASSES[oid-1]/'points.xyz');extents[i]=2*np.max(np.abs(points),axis=0)
    extent_tensor=torch.from_numpy(extents[None]).cuda()
    gt_boxes=torch.zeros(1,len(ids),5,device='cuda');dummy_poses=torch.zeros(1,len(ids),9,device='cuda')
    dummy_points=torch.zeros(1,len(ids),1,3,device='cuda');dummy_symmetry=torch.zeros(1,len(ids),device='cuda')
    initializers={};started=time.time();processed=0
    provenance=hashlib.sha256()
    for directory in ('lib','experiments/cfgs'):
        for path in sorted((a.source/directory).rglob('*')):
            if path.suffix in ('.py','.yml','.cu','.cpp'):provenance.update(str(path.relative_to(a.source)).encode());provenance.update(path.read_bytes())
    for path in (Path(__file__),root/'tools/val_non_gt_common.py',root/'tools/standalone_bop_state.py'):
        provenance.update(path.name.encode());provenance.update(path.read_bytes())
    receipt=dict(completed=False,split='val',uses_gt_pose=False,fp_calls=0,backend='released_posecnn_rgb_s0_epoch16',
        checkpoint_sha256=sha(a.checkpoint),checkpoint=str(a.checkpoint),inference_source_sha256=provenance.hexdigest(),
        split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],initializers=initializers,
        subset=bool(a.limit_streams or a.max_scan_frames),coordinate_system='original mesh to OpenCV camera; meters',
        native_build_receipt_sha256=sha(a.native_build/'build_receipt.json'),native_build=build,
        selection='Scan forward from frame0. First frame with any legal prediction for the known object, highest class confidence in that frame. No GT filtering, no future-pose borrowing, no score threshold added.',
        dummy_training_arguments='Zero tensors only for unused auxiliary target outputs of the official eval forward. No real labels, boxes, object poses or hand annotations are loaded.')
    def save():
        receipt.update(processed_rgb_frames=processed,completed_streams=len(initializers),seconds=time.time()-started,access_audit=guard.snapshot())
        temporary=a.out/'initializers.tmp';temporary.write_text(json.dumps(receipt,indent=2));temporary.replace(a.out/'initializers.json')
    with torch.no_grad(),(a.out/'attempts.jsonl').open('w') as trace:
        for stream in streams:
            found=None;K=np.asarray(stream['intrinsics'],dtype='f4');metadata=np.zeros((1,18),dtype='f4')
            metadata[0,:9]=K.reshape(-1);metadata[0,9:]=np.linalg.pinv(K).reshape(-1);metadata=torch.from_numpy(metadata).cuda()
            count=min(stream['num_frames'],a.max_scan_frames) if a.max_scan_frames else stream['num_frames']
            for frame in range(count):
                path=a.data_root/stream['relative_dir']/f'color_{frame:06d}.jpg';guard.check(path,native=True)
                image=cv2.imread(str(path));assert image is not None and image.shape==(480,640,3)
                value=torch.from_numpy(image)/255.;value-=torch.from_numpy(cfg.PIXEL_MEANS/255.).float()
                x=value.permute(2,0,1).float().contiguous()[None].cuda()
                labels=torch.zeros(1,len(ids),480,640,device='cuda')
                _,_,rois,poses,quaternions=model(x,labels,metadata,extent_tensor,gt_boxes,dummy_poses,dummy_points,dummy_symmetry)
                rois=rois.cpu().numpy();poses=poses.cpu().numpy();quaternions=quaternions.cpu().numpy();candidates=[]
                for j in nms(rois,.5):
                    cls=int(rois[j,1])
                    if cls<=0 or cls>=len(ids):continue
                    qt=quaternions[j,4*cls:4*cls+4];length=np.linalg.norm(qt)
                    if not np.isfinite(length) or length<=0:continue
                    qt=qt/length;pose=poses[j].copy();pose[4:6]*=pose[6]
                    pose[:4]=allocentric2egocentric(qt,pose[4:])
                    transform=np.eye(4);transform[:3,:3]=quat2mat(pose[:4]);transform[:3,3]=pose[4:]
                    candidates.append(dict(object_id=ids[cls],score=float(rois[j,6]),pose_original=transform.tolist()))
                selected=first_legal_for_object(candidates,stream['object_id']);processed+=1
                trace.write(json.dumps(dict(stream_id=stream['stream_id'],frame_index=frame,candidates=candidates,selected=selected is not None))+'\n');trace.flush()
                if selected is not None:
                    found=dict(frame_index=frame,pose_original=selected['pose_original'],score=selected['score'],rgb_sha256=sha(path));break
            initializers[stream['stream_id']]=found;save()
            print(json.dumps(dict(streams=len(initializers),total=len(streams),rgb_frames=processed,initialized=found is not None)),flush=True)
    assert not any(v for k,v in guard.snapshot()['counts'].items() if k.startswith('denied_'))
    receipt.update(completed=True,initialized_streams=sum(v is not None for v in initializers.values()),
        attempts_sha256=sha(a.out/'attempts.jsonl'));save()


if __name__=='__main__':main()
