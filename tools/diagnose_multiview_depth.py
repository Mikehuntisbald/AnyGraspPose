"""Bounded read-only depth audit: existing 20 physical frames across 8 views.

Audit all annotated objects, excluding hand arrays. No fitted correction is
applied. Per-pixel subsamples are retained for diagnostic model comparison only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from lip.data.index import read_yaml, CAMERAS
from lip.geometry.renderer import Renderer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    raw = root/'cache/raw_full_20260910'; index = root/'cache/dexycb_s0'
    source = root/'runs/fp_transition_analysis_21530_31000/gt_initialization_diagnostic.json'
    selected = json.loads(source.read_text())['rows']
    streams = {s['stream_id']: s for s in map(json.loads, (index/'streams.jsonl').read_text().splitlines())}
    mesh_records = {s['object_id']: s for s in streams.values()}
    meshes = {}
    for oid, s in mesh_records.items():
        with np.load(index/s['mesh_cache']) as z: meshes[oid] = {k:z[k].copy() for k in z.files}
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.cuda.set_device(0)
    renderer = Renderer('cuda'); rows=[]; worlds={}; samples=[]
    rng = np.random.default_rng(924)
    y,x = np.mgrid[:480,:640]; pixel = np.stack((x,y,np.ones_like(x)), axis=-1)
    for selection_index, chosen in enumerate(selected):
        physical = '/'.join(chosen['stream_id'].split('/')[:2]); fi=chosen['frame_index']
        meta = read_yaml(raw/physical/'meta.yml')
        ext = read_yaml(raw/'calibration'/f"extrinsics_{meta['extrinsics']}"/'extrinsics.yml')
        for ci, camera in enumerate(CAMERAS):
            sid=physical+'/'+camera; s=streams[sid]; directory=raw/sid
            depth=cv2.imread(str(directory/f'aligned_depth_to_color_{fi:06d}.png'), -1).astype('f4')*.001
            with np.load(directory/f'labels_{fi:06d}.npz') as z: seg=z['seg'].copy(); poses=z['pose_y'].copy()
            intr=read_yaml(raw/'calibration/intrinsics'/f'{camera}_640x480.yml')
            k=np.array(s['intrinsics'],dtype='f4'); rays=pixel @ np.linalg.inv(k).T
            kd=np.array([[intr['depth']['fx'],0,intr['depth']['ppx']], [0,intr['depth']['fy'],intr['depth']['ppy']], [0,0,1]],dtype='f4')
            E=np.eye(4); E[:3]=np.array(intr['extrinsics']).reshape(3,4)
            world=np.eye(4); world[:3]=np.array(ext['extrinsics'][camera]).reshape(3,4)
            for oi,oid in enumerate(meta['ycb_ids']):
                if not poses[oi].any(): continue
                pose=np.eye(4,dtype='f4');pose[:3]=poses[oi]
                key=f'{selection_index}/{oid}'; worlds.setdefault(key,[]).append(world@pose)
                if (seg==oid).sum()<100: continue
                mesh=meshes[oid]; centered=pose.copy();centered[:3,3]+=pose[:3,:3]@mesh['center']
                # Padded viewport measures truncation separately from occlusion.
                kp=k.copy(); kp[:2,2]+=[128,208]
                d,_=renderer(mesh,torch.tensor(centered,device='cuda'),torch.tensor(kp,device='cuda'),896)
                full=d[0].cpu().numpy();rd=full[208:688,128:768]
                sil=rd>0; mask=(seg==oid)&(depth>0)&sil
                interior=cv2.erode(mask.astype('uint8'),np.ones((5,5),'uint8')).astype(bool)
                if interior.sum()<100:continue
                ids=np.flatnonzero(interior); take=ids[rng.choice(len(ids), min(256,len(ids)), replace=False)]
                xyz=rays*rd[...,None]
                zE=xyz@E[2,:3]+E[2,3]; Ei=np.linalg.inv(E); zi=xyz@Ei[2,:3]+Ei[2,3]
                # Also quantify using depth intrinsics on the already aligned image.
                dd,_=renderer(mesh,torch.tensor(centered,device='cuda'),torch.tensor(kd,device='cuda'),640)
                dd=dd[0,:480].cpu().numpy(); md=interior&(dd>0)
                residual=(depth-rd)*1000; r=residual[interior]
                row=dict(selection=selection_index,object_id=int(oid),target=oid==chosen['object_id'],
                    stream_id=sid,frame_index=fi,camera=camera,extrinsics=meta['extrinsics'],subject=s['subject_id'],
                    pixels=int(interior.sum()),image_fraction=float(sil.sum()/max(1,(full>0).sum())),
                    visible_fraction_in_image=float(((seg==oid)&sil).sum()/max(1,sil.sum())),
                    u=float(np.median(x[interior])),v=float(np.median(y[interior])),gt_z_m=float(np.median(rd[interior])),
                    depth_gap_median_mm=float(np.median(r)),depth_gap_median_abs_mm=float(np.median(np.abs(r))),
                    color_to_depth_z_hypothesis_median_mm=float(np.median((zi-rd)[interior])*1000),
                    depth_to_color_z_hypothesis_median_mm=float(np.median((zE-rd)[interior])*1000),
                    depth_K_residual_median_abs_mm=float(np.median(np.abs((depth-dd)[md]))*1000) if md.any() else None)
                rows.append(row)
                points=np.column_stack((np.full(len(take),len(rows)-1),np.full(len(take),ci),rays.reshape(-1,3)[take,:2],rd.ravel()[take],depth.ravel()[take]))
                samples.append(points)
        print(json.dumps(dict(completed_selections=selection_index+1,rows=len(rows))),flush=True)
    disagreement=[float(np.max(np.abs(np.array(ws)-ws[0]))) for ws in worlds.values()]
    report=dict(completed=True,scope='20 fixed physical frame selections across all 8 cameras; all object labels only; >=100 eroded interior pixels. No pose correction or training.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_cross_camera_world_pose_element_difference=max(disagreement),rows=rows)
    (out/'multiview_depth.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'multiview_pixels.npz',values=np.concatenate(samples),columns=np.array(['row_index','camera_index','ray_x','ray_y','GT_z_m','observed_z_m']))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
