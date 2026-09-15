"""Predict train-fragment initial poses on the exact requested RGB frames."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.config import check_data_gate
from lip.engine.stream_checkpoint import sha
from lip.data.external_initializers import initializer_key
from val_non_gt_common import NativeSplitInferenceGuard,first_legal_for_object
from posecnn_backend import PoseCNNBackend


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('source','native-build','deps','checkpoint','data-root','index-root','requests','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-checkpoint-sha',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--limit',type=int);a=p.parse_args();audit=check_data_gate(a.index_root);spec=json.loads(a.requests.read_text())
    assert spec['completed'] and spec['split']=='train' and all(spec[k]==audit[k] for k in ('split_hash','mesh_hash'))
    assert 0<=a.rank<a.world
    streams=[s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines()) if s['split']=='train'];registry={s['stream_id']:s for s in streams}
    requests=spec['requests'];assert len({r['key'] for r in requests})==len(requests)==spec['unique_requests']
    for row in requests:
        s=registry[row['stream_id']];assert row['key']==initializer_key(row['stream_id'],row['frame_index'])
        assert row['object_id']==s['object_id'] and 0<=row['frame_index']<s['num_frames']
    if a.limit:requests=requests[:a.limit]
    requests=requests[a.rank::a.world];a.out.mkdir(parents=True,exist_ok=False)
    guard=NativeSplitInferenceGuard(a.data_root,a.index_root,'/mnt/why/dexycb_lip/third_party/FoundationPose',streams,'train')
    sys.addaudithook(guard);torch.set_num_threads(2);cv2.setNumThreads(0)
    backend=PoseCNNBackend(a.source,a.native_build,a.deps,a.checkpoint,a.data_root,a.expected_checkpoint_sha,a.out/'model_load.log')
    values={};start=time.time();receipt=dict(completed=False,split='train',uses_gt_pose=False,fp_calls=0,
        split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],backend='released_posecnn_rgb_s0_epoch16',
        checkpoint_sha256=sha(a.checkpoint),inference_source_sha256=backend.source_sha256,native_build=backend.native_build,
        entrypoint_sha256=sha(Path(__file__)),guard_sha256=sha(Path(__file__).parent/'val_non_gt_common.py'),
        requests_sha256=sha(a.requests),rank=a.rank,world=a.world,subset=bool(a.limit),expected_requests=len(requests),
        initializers=values,coordinate_system='original mesh to OpenCV camera; meters',
        policy='Exact requested train frame; maximum-confidence legal target-object pose. No GT filtering, no future-frame scan. Missing detector output is explicit.')
    def save():
        receipt.update(processed=len(values),seconds=time.time()-start,access_audit=guard.snapshot())
        f=a.out/'initializers.tmp';f.write_text(json.dumps(receipt,indent=2));f.replace(a.out/'initializers.json')
    with (a.out/'attempts.jsonl').open('w') as log:
        for i,row in enumerate(requests):
            s=registry[row['stream_id']];path=a.data_root/s['relative_dir']/f"color_{row['frame_index']:06d}.jpg"
            guard.check(path,native=True);image=cv2.imread(str(path));assert image is not None
            candidates=backend.predict(image,s['intrinsics']);selected=first_legal_for_object(candidates,s['object_id'])
            values[row['key']]=None if selected is None else dict(selected,stream_id=row['stream_id'],frame_index=row['frame_index'],rgb_sha256=sha(path))
            log.write(json.dumps(dict(**row,candidates=candidates,selected=selected is not None))+'\n')
            if (i+1)%128==0:log.flush();save();print(json.dumps(dict(processed=len(values),expected=len(requests))),flush=True)
    assert not any(v for k,v in guard.snapshot()['counts'].items() if k.startswith('denied_'))
    receipt.update(completed=True,attempts_sha256=sha(a.out/'attempts.jsonl'),initialized=sum(v is not None for v in values.values()));save()


if __name__=='__main__':main()
