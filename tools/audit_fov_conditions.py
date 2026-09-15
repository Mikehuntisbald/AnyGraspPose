"""Separate mesh image-boundary crossing from the existing in-image visibility."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('evaluation','index-root','data-root','out'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    m=json.loads((a.evaluation/'manifest.json').read_text());assert m['completed'] and m['population_verified'] and m['frames']==23200 and m['split']=='val'
    raw=(a.evaluation/'predictions.jsonl').read_bytes();groups={}
    for r in map(json.loads,raw.splitlines()):groups.setdefault(r['stream_id'],[]).append(r)
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())};meshes={};output=[]
    for sid,rs in sorted(groups.items()):
        s=streams[sid];rs=sorted(rs,key=lambda r:r['frame_index']);k=np.array(s['intrinsics'],dtype='f8')
        with Image.open(a.data_root/s['relative_dir']/f"color_{rs[0]['frame_index']:06d}.jpg") as im:w,h=im.size
        if s['mesh_cache'] not in meshes:
            with np.load(a.index_root/s['mesh_cache']) as z:meshes[s['mesh_cache']]=(z['vertices'].astype('f8'),z['center'].astype('f8'))
        vertices,center=meshes[s['mesh_cache']]
        with np.load(a.index_root/s['pose_cache']) as z:gt={int(f):t.astype('f8') for f,t in zip(z['frames'],z['poses'])}
        for r in rs:
            if r['initialization']:continue
            pose=gt[r['frame_index']].copy();pose[:3,3]+=pose[:3,:3]@center;camera=vertices@pose[:3,:3].T+pose[:3,3]
            front=bool((camera[:,2]>.001).all());uv=camera@k.T;xy=uv[:,:2]/np.maximum(uv[:,2:],1e-8);lo=xy.min(0);hi=xy.max(0)
            valid=front and np.isfinite(xy).all();contained=bool(valid and lo[0]>=-.5 and lo[1]>=-.5 and hi[0]<=w-.5 and hi[1]<=h-.5)
            projected_center=k@pose[:3,3];projected_center=projected_center[:2]/projected_center[2]
            center_inside=bool(valid and -.5<=projected_center[0]<=w-.5 and -.5<=projected_center[1]<=h-.5)
            output.append(dict(stream_id=sid,frame=r['frame_index'],object_id=r['object_id'],visibility=r['visibility'],
                projection_valid=bool(valid),mesh_fully_inside_image=contained,mesh_crosses_image_boundary=bool(valid and not contained),center_inside_image=center_inside,
                projected_bbox=[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],image_width=w,image_height=h,
                add_01=bool(r['add_01']),adds_005=bool(r['adds_005']),center_mm=r['center_mm']))
    assert len(output)==22880
    report=dict(completed=True,checkpoint_sha256=m['checkpoint_sha256'],prediction_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Read-only GT geometry diagnostic. Existing visibility is segmentation overlap divided by the GT rendered silhouette INSIDE the image. It does not count outside-image area in that denominator. Boundary crossing is an additional observation condition, not a replacement metric or causal failure attribution.',
        definition='All mesh vertices projected at GT pose; fully contained iff all are in front of camera and inside half-pixel image boundaries. No silhouette area fraction is claimed.',populations={})
    for name,rows in [('all',output),('visibility_lt_05',[r for r in output if r['visibility'] is not None and r['visibility']<.5]),
        ('visibility_lt_03',[r for r in output if r['visibility'] is not None and r['visibility']<.3]),('visibility_unknown',[r for r in output if r['visibility'] is None])]:
        result=dict(frames=len(rows),fully_contained=sum(r['mesh_fully_inside_image'] for r in rows),boundary_crossing=sum(r['mesh_crosses_image_boundary'] for r in rows),
            projected_center_outside=sum(r['projection_valid'] and not r['center_inside_image'] for r in rows),invalid_projection=sum(not r['projection_valid'] for r in rows),conditional_metrics={})
        for condition in ('mesh_fully_inside_image','mesh_crosses_image_boundary'):
            selected=[r for r in rows if r[condition]];objs=sorted(set(r['object_id'] for r in selected))
            result['conditional_metrics'][condition]=dict(frames=len(selected),objects=objs,
                adds005_object_macro=float(np.mean([np.mean([r['adds_005'] for r in selected if r['object_id']==o]) for o in objs])) if objs else None,
                adds005_frame_micro=float(np.mean([r['adds_005'] for r in selected])) if selected else None)
        report['populations'][name]=result
    (a.out/'fov.json').write_text(json.dumps(report,indent=2,allow_nan=False));(a.out/'frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in output));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
