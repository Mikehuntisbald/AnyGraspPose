"""Pure official FoundationPose tracking, matched to a saved LIP val manifest."""
import argparse, hashlib, json, os, sys, time, logging
from pathlib import Path
import numpy as np
import torch
import trimesh
from lip.data.index import read_frame
from lip.engine.config import check_data_gate
from lip.evaluation.metrics import errors, summarize
from lip.geometry.so3 import center_pose
from lip.integrations.foundationpose import FoundationPoseAdapter


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fp-root',required=True);p.add_argument('--data-root',default=os.getenv('DEX_YCB_DIR'))
    p.add_argument('--index-root',default='cache/dexycb_s0');p.add_argument('--reference',required=True)
    p.add_argument('--out',required=True);p.add_argument('--limit-streams',type=int);p.add_argument('--max-frames',type=int)
    a=p.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'predictions.jsonl').exists():raise RuntimeError('Use a fresh output directory; never overwrite previous predictions')
    fp_root=Path(a.fp_root).resolve();sys.path.insert(0,str(fp_root))
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    import nvdiffrast.torch as dr
    # Registration rotation hypotheses and scorer are unused for GT-initialized tracking.
    # Keep the official reset_object and track_one implementations unchanged.
    class TrackingOnly(FoundationPose):
        def make_rotation_grid(self,*args,**kwargs):pass
    logging.getLogger().setLevel(logging.WARNING)
    np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    index=Path(a.index_root);ref=Path(a.reference);root=Path(a.data_root)
    audit=check_data_gate(index);manifest=json.loads((ref/'manifest.json').read_text())
    assert manifest['mode']=='closed-loop' and manifest['split']=='val'
    assert manifest['initial_pose_source']=='gt_first_frame' and not manifest['quick_subset']
    assert manifest['split_hash']==audit['split_hash'] and manifest['mesh_hash']==audit['mesh_hash']
    all_streams={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines())}
    ids=manifest['streams'][:a.limit_streams];reference={}
    with (ref/'predictions.jsonl').open() as f:
        for line in f:
            r=json.loads(line);reference[(r['stream_id'],r['frame_index'])]=r
    weights={str(f.relative_to(fp_root)):sha(f) for f in sorted((fp_root/'weights').rglob('*')) if f.is_file()}
    run=dict(method='pure FoundationPose',mode='closed-loop',split='val',initial_pose_source='gt_first_frame',
             iteration=2,registration=False,post_initial_gt_feedback=False,streams=ids,
             full_sequences=a.max_frames is None,quick_subset=a.limit_streams is not None,
             split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],weights_sha256=weights,
             reference_manifest_sha256=sha(ref/'manifest.json'),reference_predictions_sha256=sha(ref/'predictions.jsonl'),
             precision='official refiner AMP float16',seed=42,compatibility='diameter cast to Python float; no numeric value change',
             gt_usage='first frame initialization; subsequent labels used only after prediction for metrics',
             initialization_adapter='skip unused registration rotation grid and scorer; official reset_object/track_one unchanged')
    import subprocess
    run['foundationpose_commit']=subprocess.check_output(['git','-c','safe.directory='+str(fp_root),'-C',str(fp_root),'rev-parse','HEAD'],text=True).strip()
    run['evaluator_sha256']=sha(__file__)
    (out/'manifest.json').write_text(json.dumps(run,indent=2))
    refiner=PoseRefinePredictor();glctx=dr.RasterizeCudaContext();est=None;last_oid=None
    rows=[];latencies=[];first_failure={};started=time.time()
    with (out/'predictions.jsonl').open('w') as f:
        for si,sid in enumerate(ids):
            s=all_streams[sid];assert s['split']=='val'
            with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            with np.load(index/s['pose_cache']) as z:frames=z['frames'].copy();original=z['poses'].copy()
            frames=frames[:a.max_frames];K=np.array(s['intrinsics'],dtype=np.float32)
            if last_oid!=s['object_id']:
                raw=trimesh.load(root/s['mesh_path'],process=False,force='mesh');raw.vertices*=audit['mesh_scale_to_m']
                assert np.allclose((raw.vertices.max(0)+raw.vertices.min(0))/2,mesh['center'],atol=1e-6)
                if est is None:est=TrackingOnly(raw.vertices,raw.vertex_normals,mesh=raw,scorer=object(),refiner=refiner,glctx=glctx,debug=0,debug_dir=str(out/'fp_debug'))
                else:est.reset_object(raw.vertices,raw.vertex_normals,mesh=raw)
                est.diameter=float(est.diameter)  # Avoid NumPy scalar promotion in official crop tensor construction.
                last_oid=s['object_id']
            adapter=FoundationPoseAdapter(est);prior=original[0].copy();adapter.accept(prior);streak=0
            for j,frame in enumerate(frames):
                torch.cuda.synchronize();begin=time.perf_counter()
                rgb,depth=read_frame(root,s,int(frame),audit['depth_scale_to_m']);rgb=np.rint(rgb.transpose(1,2,0)*255).astype('uint8');depth=depth[0]
                assert depth.ndim==2 and rgb.shape[:2]==depth.shape
                torch.cuda.synchronize();decoded=time.perf_counter();nonfinite=False
                if j:
                    pred=est.track_one(rgb=rgb,depth=depth,K=K,iteration=2)
                    if not np.isfinite(pred).all():raise RuntimeError(f'Nonfinite official output: {sid} {frame}')
                else:pred=prior.copy()
                prior=np.array(pred,copy=True)
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                # No GT or reference output is passed into track_one after initialization.
                centered=center_pose(torch.from_numpy(prior),torch.from_numpy(mesh['center']))
                gt=center_pose(torch.from_numpy(original[j]),torch.from_numpy(mesh['center']))
                r=reference[(sid,int(frame))]
                e=errors(centered,gt,mesh['vertices'],float(mesh['diameter']))
                streak=streak+1 if e['adds_01']==0 else 0
                if streak==5 and sid not in first_failure:first_failure[sid]=int(frame)
                row=dict(**e,object_id=s['object_id'],stream_id=sid,frame_index=int(frame),visibility=r['visibility'],
                         visibility_bin=r['visibility_bin'],moving=r['moving'],lost=streak>=5,nonfinite_output=nonfinite,
                         pose_centered=centered.tolist(),pose_original=prior.tolist())
                f.write(json.dumps(row)+'\n');f.flush();rows.append(row)
                if j:latencies.append(dict(preprocess=decoded-begin,end_to_end=elapsed))
            print(json.dumps(dict(streams=si+1,total_streams=len(ids),frames=len(rows),elapsed=time.time()-started)),flush=True)
    expected={(sid,int(frame)) for sid in ids for frame in np.load(index/all_streams[sid]['pose_cache'])['frames'][:a.max_frames]}
    assert {(r['stream_id'],r['frame_index']) for r in rows}==expected and len(rows)==len(expected)
    if a.limit_streams is None and a.max_frames is None:assert expected==set(reference)
    report=summarize(rows);report['latency_seconds']={k:dict(mean=float(np.mean([x[k] for x in latencies])),p95=float(np.quantile([x[k] for x in latencies],.95))) for k in latencies[0]} if latencies else {}
    report['latency_scope']='batch=1 synchronized; includes raw decode and official depth preprocessing/refinement; excludes GT metrics and object setup'
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    run.update(completed=True,frames=len(rows),first_five_consecutive_adds_failures=first_failure)
    (out/'manifest.json').write_text(json.dumps(run,indent=2))
    print(json.dumps(dict(completed=True,macro_object=report['macro_object'])),flush=True)

if __name__=='__main__':main()
