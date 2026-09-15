"""Complete train-only visibility using existing endpoint statistics plus raw audits."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack_legacy(streams,pose_frames,clips,length=8):
    values=[[None]*len(f) for f in pose_frames];known=[[False]*len(f) for f in pose_frames]
    positions=[{int(f):i for i,f in enumerate(fs)} for fs in pose_frames]
    for item in clips:
        sid=item['stream'];end=item['start']+(length-1)*item['stride'];pos=positions[sid][end];v=item['visibility']
        if v is not None and (not np.isfinite(v) or not 0<=v<=1):raise ValueError('Invalid cached visibility')
        if known[sid][pos] and values[sid][pos]!=v:raise ValueError('Contradictory cached endpoint')
        values[sid][pos]=v;known[sid][pos]=True
    return values,known


def bbox_inside(mesh,poses,k):
    """Conservative mesh-box test, not exact rendered silhouette coverage."""
    v=mesh['vertices'];lo=v.min(0);hi=v.max(0);corners=np.array(list(itertools.product(*zip(lo,hi))))
    camera=corners[None]@poses[:,:3,:3].transpose(0,2,1)+poses[:,None,:3,3]
    uv=camera@np.asarray(k).T;xy=uv[...,:2]/np.maximum(uv[...,2:],1e-8)
    return ((camera[...,2]>.001).all(1)&(xy[...,0]>=-.5).all(1)&(xy[...,0]<=639.5).all(1)&(xy[...,1]>=-.5).all(1)&(xy[...,1]<=479.5).all(1)).tolist()


def prepare(a):
    import torch
    from lip.geometry.so3 import center_pose
    from lip.engine.config import check_data_gate
    a.out.mkdir(parents=True,exist_ok=False);audit=check_data_gate(a.index_root)
    streams=[s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines()) if s['split']=='train']
    frames=[];poses=[];meshes={}
    for s in streams:
        assert sha(a.index_root/s['pose_cache'])==s['pose_cache_sha256']
        with np.load(a.index_root/s['pose_cache']) as z:frames.append(z['frames'].copy());poses.append(z['poses'].copy())
        if s['mesh_cache'] not in meshes:
            with np.load(a.index_root/s['mesh_cache']) as z:meshes[s['mesh_cache']]={k:z[k].copy() for k in z.files}
    pool=a.index_root/'train_clips_L8.json';clips=json.loads(pool.read_text())
    shard_paths=sorted((a.index_root/'sampling_shards').glob('train_L8_rank*.json'));shard_hashes={}
    for path in shard_paths:
        d=json.loads(path.read_text());assert d['split']=='train' and d['split_hash']==audit['split_hash'];shard_hashes[path.name]=sha(path)
    values,known=unpack_legacy(streams,frames,clips);del clips
    rng=np.random.default_rng(20260914);groups={};pairs=[]
    for sid,s in enumerate(streams):
        valid=np.flatnonzero(known[sid]);assert len(valid)>0
        groups.setdefault((s['subject_id'],s['object_id'],s['camera_serial']),[]).extend((sid,int(i)) for i in valid)
        pairs.extend((sid,int(i)) for i in valid)
    audits={group[int(rng.integers(len(group)))] for group in groups.values()}
    audits.update(pairs[int(i)] for i in rng.choice(len(pairs),512,replace=False))
    records=[]
    for sid,s in enumerate(streams):
        mesh=meshes[s['mesh_cache']];centered=center_pose(torch.from_numpy(poses[sid]),torch.from_numpy(mesh['center'])).numpy()
        missing=[int(frames[sid][i]) for i,v in enumerate(known[sid]) if not v]
        audit_frames=[int(frames[sid][i]) for i in range(len(frames[sid])) if (sid,i) in audits]
        records.append(dict(stream_id=s['stream_id'],object_id=s['object_id'],frames=frames[sid].tolist(),values=values[sid],known=known[sid],
            missing_frames=missing,audit_frames=audit_frames,bbox_fully_inside_image=bbox_inside(mesh,centered,s['intrinsics'])))
    base=a.out/'base.jsonl';base.write_text(''.join(json.dumps(r)+'\n' for r in records))
    manifest=dict(prepared=True,split='train',streams=len(streams),frames=sum(len(x) for x in frames),world=a.world,
        data_root=str(a.data_root.resolve()),index_root=str(a.index_root.resolve()),runtime=str(a.runtime.resolve()),
        split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],stream_index_sha256=sha(a.index_root/'streams.jsonl'),legacy_pool_sha256=sha(pool),legacy_shards_sha256=shard_hashes,
        collector_sha256=sha(__file__),renderer_sha256=sha(a.runtime/'src/lip/geometry/renderer.py'),base_sha256=sha(base),
        reused_frame_values=sum(sum(k) for k in known),missing_frame_values=sum(len(r['missing_frames']) for r in records),raw_audit_frames=len(audits),
        audit_design='One cached frame from every subject/object/camera cell plus 512 uniform cached frames, fixed seed; overlap removed.',
        provenance_limit='Legacy sampling shards record split hash but no mesh/code hash. Reuse is conditional on exact raw checks; this is not a raw recomputation of every cached frame.',
        semantics='Target object mask overlap / GT rendered silhouette inside the image, same as lip.evaluate.visibility. Only object mask is used; no hand pose or hand-class supervision. Train-only labels for diagnostics/sampling, never predictor inputs.',
        bbox_semantics='All eight corners of the centered CAD axis-aligned box inside the image; conservative full-object containment test, not exact silhouette area.')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


def worker(a):
    import torch
    from lip.geometry.so3 import center_pose
    from lip.geometry.renderer import Renderer
    from lip.evaluate import visibility as reference_visibility
    torch.set_num_threads(1);torch.cuda.set_device(0)
    m=json.loads((a.out/'manifest.json').read_text());assert m['collector_sha256']==sha(__file__) and m['base_sha256']==sha(a.out/'base.jsonl')
    assert m['renderer_sha256']==sha(a.runtime/'src/lip/geometry/renderer.py') and m['stream_index_sha256']==sha(a.index_root/'streams.jsonl')
    all_records=list(map(json.loads,(a.out/'base.jsonl').read_text().splitlines()));records=all_records[a.rank::m['world']]
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())};renderer=Renderer('cuda');meshes={};frames_done=0;audited=0;mismatches=[];started=time.time();reference_checks=0
    dest=a.out/f'rank{a.rank}';dest.mkdir(exist_ok=False)
    with torch.no_grad(),(dest/'values.jsonl').open('w') as writer:
        for r in records:
            s=streams[r['stream_id']];assert s['split']=='train'
            if s['mesh_cache'] not in meshes:
                with np.load(a.index_root/s['mesh_cache']) as z:meshes[s['mesh_cache']]={k:z[k].copy() for k in z.files}
            mesh=meshes[s['mesh_cache']]
            with np.load(a.index_root/s['pose_cache']) as z:indices=z['frames'].copy();centered=center_pose(torch.from_numpy(z['poses']),torch.from_numpy(mesh['center']))
            mapping={int(f):i for i,f in enumerate(indices)};k=torch.tensor(s['intrinsics'],device='cuda');requested=sorted(set(r['missing_frames']+r['audit_frames']));result=[]
            for frame in requested:
                path=a.data_root/s['relative_dir']/f'labels_{frame:06d}.npz'
                with np.load(path,allow_pickle=False) as z:
                    target=(z['seg']==s['object_id']) if 'seg' in z else None
                gt=centered[mapping[frame]].cuda();visible=total=0
                if target is not None:
                    assert target.shape==(480,640)
                    depth,_=renderer(mesh,gt,k,640);mask=depth[0,:480,:640]>0
                    total=int(mask.sum());visible=int((mask&torch.from_numpy(target).cuda()).sum());value=visible/total if total else None
                else:value=None
                if frame in r['audit_frames']:
                    audited+=1;old=r['values'][mapping[frame]]
                    if value!=old:mismatches.append(dict(stream_id=r['stream_id'],frame=frame,cached=old,recomputed=value))
                    if reference_checks<2:
                        assert value==reference_visibility(a.data_root,s,frame,gt,k,mesh,renderer);reference_checks+=1
                result.append(dict(frame=frame,visibility=value,visible_pixels=visible,projected_pixels=total,missing_seg=target is None));frames_done+=1
            writer.write(json.dumps(dict(stream_id=r['stream_id'],values=result))+'\n');writer.flush()
            if frames_done%1000<20:print(json.dumps(dict(rank=a.rank,frames=frames_done,seconds=time.time()-started)),flush=True)
    receipt=dict(completed=not mismatches,rank=a.rank,world=m['world'],streams=len(records),raw_frames=frames_done,audited_frames=audited,
        exact_reference_function_checks=reference_checks,cached_mismatches=mismatches,seconds=time.time()-started,values_sha256=sha(dest/'values.jsonl'),collector_sha256=sha(__file__))
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='cached_mismatches'}),flush=True)
    if mismatches:raise RuntimeError('Legacy visibility values differ from raw recomputation; reuse rejected')


def merge(a):
    m=json.loads((a.out/'manifest.json').read_text());assert m['collector_sha256']==sha(__file__) and m['base_sha256']==sha(a.out/'base.jsonl')
    records=list(map(json.loads,(a.out/'base.jsonl').read_text().splitlines()));byid={r['stream_id']:r for r in records};seen=set();receipts=[]
    for rank in range(m['world']):
        dest=a.out/f'rank{rank}';receipt=json.loads((dest/'receipt.json').read_text());assert receipt['completed'] and receipt['rank']==rank and receipt['world']==m['world'] and receipt['collector_sha256']==sha(__file__)
        assert receipt['values_sha256']==sha(dest/'values.jsonl');receipts.append(receipt)
        for result in map(json.loads,(dest/'values.jsonl').read_text().splitlines()):
            sid=result['stream_id'];assert sid not in seen;seen.add(sid);r=byid[sid];positions={f:i for i,f in enumerate(r['frames'])}
            assert sorted(v['frame'] for v in result['values'])==sorted(set(r['missing_frames']+r['audit_frames']))
            for v in result['values']:
                pos=positions[v['frame']]
                if r['known'][pos]:assert r['values'][pos]==v['visibility']
                r['values'][pos]=v['visibility'];r['known'][pos]=True
    assert seen==set(byid) and all(all(r['known']) for r in records)
    assert sum(len(r['frames']) for r in records)==m['frames'] and sum(x['raw_frames'] for x in receipts)==m['missing_frame_values']+m['raw_audit_frames']
    output=a.out/'train_visibility.jsonl';output.write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='known'})+'\n' for r in sorted(records,key=lambda x:x['stream_id'])))
    report=dict(m,completed=True,output_sha256=sha(output),all_frame_positions_covered=True,raw_frames_recomputed=sum(x['raw_frames'] for x in receipts),
        cached_values_audited=sum(x['audited_frames'] for x in receipts),cached_mismatches=0,worker_receipts=receipts)
    (a.out/'completed.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='worker_receipts'},indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','worker','merge'))
    for key in ('runtime','data-root','index-root','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--world',type=int,default=8);p.add_argument('--rank',type=int,default=0);a=p.parse_args();sys.path.insert(0,str(a.runtime/'src'))
    if a.world<1 or not 0<=a.rank<a.world:p.error('Invalid shard configuration')
    {'prepare':prepare,'worker':worker,'merge':merge}[a.mode](a)


if __name__=='__main__':main()
