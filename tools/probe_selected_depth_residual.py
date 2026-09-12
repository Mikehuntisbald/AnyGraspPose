"""Check observed depth against GT/FP renders on the existing 20 diagnostic frames."""
import json,os
from pathlib import Path
import cv2
import numpy as np
import torch
from lip.geometry.renderer import Renderer

R=Path('/mnt/why/dexycb_lip');os.chdir(R);out=R/'runs/fp_transition_analysis_21530_31000'
probe=json.loads((out/'gt_initialization_diagnostic.json').read_text());index=R/'cache/dexycb_s0';root=R/'cache/raw_full_20260910'
streams={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines())}
torch.cuda.set_device(0);torch.set_num_threads(2);renderer=Renderer('cuda');rows=[]
def stats(a):
    return dict(count=len(a),mean_mm=float(a.mean()),median_mm=float(np.median(a)),median_abs_mm=float(np.median(np.abs(a))))
for r in probe['rows']:
    s=streams[r['stream_id']];directory=root/s['relative_dir'];frame=r['frame_index']
    depth=cv2.imread(str(directory/f'aligned_depth_to_color_{frame:06d}.png'),cv2.IMREAD_UNCHANGED).astype('f4')*.001
    with np.load(directory/f'labels_{frame:06d}.npz') as z:target=z['seg']==s['object_id']
    with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
    k=torch.tensor(s['intrinsics'],device='cuda');renders=[]
    for name in ('input_GT_pose','refined_pose'):
        rd,_=renderer(mesh,torch.tensor(r[name],device='cuda'),k,640);renders.append(rd[0,:480,:640].cpu().numpy())
    mask=target & (depth>0) & (renders[0]>0)
    inside=cv2.erode(mask.astype('uint8'),np.ones((5,5),dtype='uint8')).astype(bool)
    common=inside & (renders[1]>0);assert common.sum()>20
    before=(depth-renders[0])*1000;after=(depth-renders[1])*1000
    rows.append(dict(object_id=s['object_id'],stream_id=s['stream_id'],frame_index=frame,
        fp_center_error_mm=r['after']['center_mm'],fp_z_shift_mm=r['signed_error_xyz_mm'][2],
        GT_target_interior=stats(before[inside]),common_before=stats(before[common]),common_after=stats(after[common]),
        common_coverage=float(common.sum()/inside.sum())))
report=dict(scope='Oracle diagnostic only, same 20 frames; observed depth minus project CUDA-rendered depth in mm. Object GT segmentation used only for diagnostic pixels; 5x5 erosion removes boundaries. Post-FP comparison uses the same pixels with valid depth in both renders.',rows=rows,
    median_GT_interior_signed_residual_mm=float(np.median([r['GT_target_interior']['median_mm'] for r in rows])),
    median_GT_interior_absolute_residual_mm=float(np.median([r['GT_target_interior']['median_abs_mm'] for r in rows])),
    mean_common_before_median_abs_mm=float(np.mean([r['common_before']['median_abs_mm'] for r in rows])),
    mean_common_after_median_abs_mm=float(np.mean([r['common_after']['median_abs_mm'] for r in rows])))
(out/'selected_depth_residual.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
