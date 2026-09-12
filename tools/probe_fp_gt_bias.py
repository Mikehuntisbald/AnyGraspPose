"""Oracle diagnostic only: frozen FP from exact object GT on clear static frames."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from lip.data.index import read_frame
from lip.geometry.so3 import center_pose,original_pose
from lip.integrations.frozen_fp import FrozenFoundationPose
from lip.integrations.foundationpose import FoundationPoseAdapter
from lip.evaluation.metrics import errors

R=Path('/mnt/why/dexycb_lip');os.chdir(R);out=R/'runs/fp_transition_analysis_21530_31000'
torch.set_num_threads(2);torch.cuda.set_device(0)
index=R/'cache/dexycb_s0';root=R/'cache/raw_full_20260910'
streams={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines())}
groups={}
for r in map(json.loads,(R/'runs/lip_fp_31000/lip/predictions.jsonl').read_text().splitlines()):
    if r['visibility'] is not None and r['visibility']>=.9 and not r['moving'] and r['frame_index']>0:
        groups.setdefault(r['object_id'],[]).append(r)
rng=np.random.default_rng(42);selected=[]
for oid in sorted(groups):
    pool=sorted(groups[oid],key=lambda r:(r['stream_id'],r['frame_index']))
    selected.append(pool[int(rng.integers(len(pool)))])
assert len(selected)==20
fp=FrozenFoundationPose(R/'third_party/FoundationPose',root,torch.device('cuda',0),
                       '774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60')
rows=[]
for r in selected:
    s=streams[r['stream_id']]
    with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
    with np.load(index/s['pose_cache']) as z:
        loc=int(np.flatnonzero(z['frames']==r['frame_index'])[0]);gt0=torch.from_numpy(z['poses'][loc].copy())
    center=torch.from_numpy(mesh['center']);gt=center_pose(gt0,center).cuda()
    rgb,depth=read_frame(root,s,r['frame_index'],.001);k=torch.tensor(s['intrinsics'])
    refined=fp(gt.detach(),torch.from_numpy(rgb),torch.from_numpy(depth),k,s['mesh_path'],mesh['center'])
    est=fp.pool[s['mesh_path']];adapter=FoundationPoseAdapter(est,'cuda:0')
    original=original_pose(gt,center.cuda());adapter.accept(original)
    inner_error=float((est.pose_last-gt).abs().max())
    back=center_pose(est.pose_last@est.get_tf_to_centered_mesh(),center.cuda())
    roundtrip=float((back-gt).abs().max());assert inner_error<1e-6 and roundtrip<1e-6
    before=errors(gt.cpu(),gt.cpu(),mesh['vertices'],float(mesh['diameter']))
    after=errors(refined.cpu(),gt.cpu(),mesh['vertices'],float(mesh['diameter']))
    row=dict(object_id=r['object_id'],mesh_path=s['mesh_path'],stream_id=r['stream_id'],frame_index=r['frame_index'],
        visibility=r['visibility'],moving=r['moving'],diameter_m=float(mesh['diameter']),before=before,after=after,
        signed_error_xyz_mm=((refined[:3,3]-gt[:3,3])*1000).cpu().tolist(),
        input_GT_pose=gt.cpu().tolist(),refined_pose=refined.cpu().tolist(),inner_pose_max_abs=inner_error,roundtrip_max_abs=roundtrip)
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('input_GT_pose','refined_pose')}),flush=True)
report=dict(completed=True,frames=len(rows),objects=len(groups),GT_used_as_input=True,
    scope='Oracle diagnostic, not a tracking benchmark. One frame per val object, seed 42, visibility >= .9 and not moving; selection does not use pose errors. No model training.',
    fp_iterations=2,fp_refiner_sha256=fp.weight_sha256,
    mean_after={k:float(np.mean([r['after'][k] for r in rows])) for k in rows[0]['after']},
    mean_signed_error_xyz_mm=np.mean([r['signed_error_xyz_mm'] for r in rows],axis=0).tolist(),
    positive_z_frames=sum(r['signed_error_xyz_mm'][2]>0 for r in rows),
    max_roundtrip_abs=max(r['roundtrip_max_abs'] for r in rows),rows=rows)
(out/'gt_initialization_diagnostic.json').write_text(json.dumps(report,indent=2))
