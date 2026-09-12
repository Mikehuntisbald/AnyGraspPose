"""Oracle causal diagnostic: replace only visible target depth with GT render.

This uses GT to construct an input and is NOT a deployable method or benchmark.
RGB, K, input pose, FP weights and all other depth pixels remain paired.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from lip.integrations.frozen_fp import FrozenFoundationPose
from lip.evaluation.metrics import errors


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    raw=root/'cache/raw_full_20260910';index=root/'cache/dexycb_s0'
    source=root/'runs/fp_transition_analysis_21530_31000/gt_initialization_diagnostic.json'
    selected=json.loads(source.read_text())['rows']
    streams={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines())}
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.cuda.set_device(0)
    fp=FrozenFoundationPose(root/'third_party/FoundationPose',raw,torch.device('cuda',0),'774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60')
    rows=[]
    for r in selected:
        s=streams[r['stream_id']]
        with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
        with np.load(out/f"object_{s['object_id']:02d}.npz") as z:
            rgb=z['rgb'].copy();depth=z['observed_depth'].copy();rd=z['project_depth'].copy();target=z['target'].copy()
        rgb=torch.from_numpy(rgb.transpose(2,0,1).astype('f4')/255)
        k=torch.tensor(s['intrinsics']);gt=torch.tensor(r['input_GT_pose'],device='cuda')
        replacement=target&(rd>0)&(depth>0)
        modified=depth.copy();modified[replacement]=rd[replacement]
        results={}
        for name,d in [('observed',depth),('oracle_target_depth',modified)]:
            after=fp(gt.detach(),rgb,torch.from_numpy(d[None]),k,s['mesh_path'],mesh['center'])
            results[name]=dict(pose_centered=after.cpu().tolist(),errors=errors(after.cpu(),gt.cpu(),mesh['vertices'],float(mesh['diameter'])),
                signed_xyz_mm=((after[:3,3]-gt[:3,3])*1000).cpu().tolist())
        old=np.array(r['refined_pose']);new=np.array(results['observed']['pose_centered'])
        row=dict(object_id=s['object_id'],stream_id=s['stream_id'],frame_index=r['frame_index'],replaced_pixels=int(replacement.sum()),
            repeated_real_FP_pose_max_abs=float(np.abs(old-new).max()),**results)
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(completed=True,scope='Oracle paired input intervention on same 20 frames. Start from GT pose in both arms. Only depth at visible target GT-segmentation pixels is replaced by GT-render depth (missing depth stays missing). No hand arrays. No training. Not a benchmark or a deployable correction.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fp_backend_file=inspect.getfile(FrozenFoundationPose),fp_backend_sha256=hashlib.sha256(Path(inspect.getfile(FrozenFoundationPose)).read_bytes()).hexdigest(),
        fp_weight_sha256=fp.weight_sha256,rows=rows)
    (out/'fp_depth_intervention.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
