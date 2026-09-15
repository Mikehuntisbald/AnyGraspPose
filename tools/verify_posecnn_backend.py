"""Check official preprocessing/pose conversion on the same real network outputs."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha
from val_non_gt_common import val_streams,NativeSplitInferenceGuard
from posecnn_backend import PoseCNNBackend


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('source','native-build','deps','checkpoint','data-root','index-root','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-checkpoint-sha',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    streams=val_streams(a.index_root)[:2];guard=NativeSplitInferenceGuard(a.data_root,a.index_root,'/mnt/why/dexycb_lip/third_party/FoundationPose',streams,'val')
    sys.addaudithook(guard);torch.set_num_threads(2);cv2.setNumThreads(0)
    backend=PoseCNNBackend(a.source,a.native_build,a.deps,a.checkpoint,a.data_root,a.expected_checkpoint_sha,a.out/'model.log');results=[]
    from fcn.config import cfg
    for s in streams:
        path=a.data_root/s['relative_dir']/'color_000000.jpg';guard.check(path,native=True);image=cv2.imread(str(path));assert image is not None
        captured={}
        def before(model,args):captured['input']=args[0].detach().cpu().clone()
        def after(model,args,output):captured['outputs']=[x.detach().cpu().numpy().copy() for x in output[2:]]
        h1=backend.model.register_forward_pre_hook(before);h2=backend.model.register_forward_hook(after)
        candidates=backend.predict(image,s['intrinsics']);h1.remove();h2.remove()
        expected=torch.from_numpy(image)/255.;expected-=torch.from_numpy(cfg.PIXEL_MEANS/255.).float()
        expected=expected.permute(2,0,1).float().contiguous()[None];assert torch.equal(expected,captured['input'])
        rois,poses,quaternions=captured['outputs'];converted={}
        # Replay the original test_dataset.py order: convert every pose, then NMS.
        for j in range(len(rois)):
            cls=int(rois[j,1])
            if not 0<cls<len(backend.ids):continue
            qt=quaternions[j,4*cls:4*cls+4];length=np.linalg.norm(qt)
            if not np.isfinite(length) or length<=0:continue
            qt=qt/length;poses[j,4]*=poses[j,6];poses[j,5]*=poses[j,6]
            poses[j,:4]=backend.allocentric(qt,poses[j,4:]);T=np.eye(4);T[:3,:3]=backend.quat2mat(poses[j,:4]);T[:3,3]=poses[j,4:]
            converted[j]=dict(object_id=backend.ids[cls],score=float(rois[j,6]),pose_original=T.tolist())
        replay=[converted[j] for j in backend.nms(rois,.5) if j in converted];assert candidates==replay
        assert all(torch.count_nonzero(t)==0 for t in (backend.labels,backend.boxes,backend.poses,backend.points,backend.symmetry))
        results.append(dict(stream_id=s['stream_id'],rgb_sha256=sha(path),input_bitwise_equal=True,conversion_bitwise_equal=True,candidates=len(candidates),zero_auxiliary_arguments_preserved=True))
    assert not any(v for k,v in guard.snapshot()['counts'].items() if k.startswith('denied_'))
    (a.out/'receipt.json').write_text(json.dumps(dict(passed=True,results=results,backend_source_sha256=backend.source_sha256,access_audit=guard.snapshot(),
        scope='Real-network preprocessing and original postprocessing parity on identical captured outputs; not a claim of bitwise repeatability across separate CUDA Hough-voting executions.'),indent=2));print('input and original conversion parity passed')


if __name__=='__main__':main()
