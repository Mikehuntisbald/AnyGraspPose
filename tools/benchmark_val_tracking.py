"""Isolated batch-one native RGB-D-to-pose latency; no frame IO in the timer."""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.config import check_data_gate
from lip.engine.stream_checkpoint import sha, source_hash
from val_non_gt_common import val_streams, validate_initializers, NativeSplitInferenceGuard


def selected_streams(streams, initial):
    """One fixed stream per object, chosen without GT/error or candidate access."""
    selected = {}
    for s in sorted(streams, key=lambda s:s['stream_id']):
        prior = initial['initializers'][s['stream_id']]
        if prior is not None and s['num_frames']-prior['frame_index']-1 >= 32:
            selected.setdefault(s['object_id'], s)
    assert len(selected) == 20
    return list(selected.values())


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--method', choices=('lip','fp'), required=True)
    for n in ('initializers','data-root','index-root','out','fp-root'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--startup-iterations',type=int,choices=(1,2),default=1);p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--expected-sha',required=True)
    a = p.parse_args(); audit = check_data_gate(a.index_root); streams = val_streams(a.index_root)
    initial = validate_initializers(json.loads(a.initializers.read_text()),streams,audit); streams = selected_streams(streams,initial)
    # Every timing worker is a separate process; FP may change default tensor type.
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active: raise RuntimeError('Isolated benchmark requires idle GPUs: '+active)
    a.out.mkdir(parents=True,exist_ok=False)
    guard = NativeSplitInferenceGuard(a.data_root,a.index_root,a.fp_root,streams,'val',allow_foundationpose=a.method=='fp')
    sys.addaudithook(guard);sys.dont_write_bytecode=True
    torch.set_num_threads(2);cv2.setNumThreads(0);torch.manual_seed(42);np.random.seed(42)
    rows=[];memory=[];pool={};calls=0
    if a.method=='lip':
        from lip.engine.stream_config import load_stream_config,make_model
        from lip.engine.stream_checkpoint import load_init
        from lip.geometry.renderer import Renderer
        from standalone_bop_state import prime_initial_observation
        from streaming_bop_utils import hold_failed_step
        c=load_stream_config(a.config);assert sha(a.checkpoint)==a.expected_sha
        model=make_model(c).cuda().eval();load_init(a.checkpoint,model,audit,c);renderer=Renderer('cuda')
        def update(state,rgb,depth,timestamp):
            from lip.engine.intraframe import step_iterated
            iterations=a.startup_iterations if 1<=state.frame_id<=8 else 1
            proposal,next_state=step_iterated(model,torch.from_numpy(rgb.transpose(2,0,1).copy()),torch.from_numpy(depth[None]),timestamp,state,iterations=iterations,
                renderer=renderer,precision=c['precision'],image_size=c['image_size'],crop_expansion=c['crop_expansion'])
            if proposal['status']=='ok':state=model.commit(proposal,next_state)
            else:state=hold_failed_step(state,timestamp)
            return state,proposal.get('pose_original'),proposal['status'],0
    else:
        import trimesh
        from lip.integrations.foundationpose import FoundationPoseAdapter
        from streaming_bop_utils import legal_pose
        sys.path.insert(0,str(a.fp_root))
        from estimater import FoundationPose
        from learning.training.predict_pose_refine import PoseRefinePredictor
        import nvdiffrast.torch as dr
        class TrackingOnly(FoundationPose):
            def make_rotation_grid(self,*args,**kwargs):pass
        logging.getLogger().setLevel(logging.WARNING)
        assert sha(a.fp_root/'weights/2023-10-28-18-33-37/model_best.pth')==a.expected_sha
        refiner=PoseRefinePredictor();refiner.model.eval().requires_grad_(False);context=dr.RasterizeCudaContext()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for s in streams:
            sid=s['stream_id'];external=initial['initializers'][sid];first=external['frame_index'];K=np.asarray(s['intrinsics'],dtype='f4')
            with np.load(a.index_root/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            if a.method=='fp':
                raw=trimesh.load(a.data_root/s['mesh_path'],process=False,force='mesh');raw.vertices*=audit['mesh_scale_to_m']
                rng=np.random.get_state();np.random.seed(42)
                try:est=TrackingOnly(raw.vertices,raw.vertex_normals,mesh=raw,scorer=object(),refiner=refiner,glctx=context,debug=0,debug_dir=str(a.out/'fp_debug'))
                finally:np.random.set_state(rng)
                est.diameter=float(est.diameter);pool[s['object_id']]=est;adapter=FoundationPoseAdapter(est,'cuda')
            for frame in range(first,first+33):
                rgbpath=a.data_root/s['relative_dir']/f'color_{frame:06d}.jpg'
                depthpath=a.data_root/s['relative_dir']/f'aligned_depth_to_color_{frame:06d}.png'
                guard.check(rgbpath,native=True);guard.check(depthpath,native=True)
                image=cv2.imread(str(rgbpath));dep=cv2.imread(str(depthpath),-1);assert image is not None and dep is not None
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);depth=dep.astype('f4')*audit['depth_scale_to_m'];timestamp=frame/audit['fps']
                if frame==first:
                    if a.method=='lip':
                        state,_=prime_initial_observation(model,np.asarray(external['pose_original'],dtype='f4'),mesh,K,sid,timestamp,
                            audit['fps'],depth.shape,s['object_id'],s['camera_serial'],lambda state:update(state,rgb,depth,timestamp))
                    else:prior=np.asarray(external['pose_original'],dtype='f4');adapter.accept(prior)
                    continue
                torch.cuda.synchronize();start=time.perf_counter()
                if a.method=='lip':
                    state,output,status,_=update(state,rgb,depth,timestamp)
                    output=output.detach().cpu().numpy()
                else:
                    proposal=np.asarray(est.track_one(rgb=rgb,depth=depth,K=K,iteration=2),dtype='f4');calls+=1
                    if legal_pose(proposal):prior=proposal;status='ok'
                    else:adapter.accept(prior);status='invalid_fp_pose_held'
                    output=prior
                assert output.shape==(4,4)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                rows.append(dict(stream_id=sid,object_id=s['object_id'],frame_index=frame,update=frame-first,seconds=elapsed,status=status))
            memory.append(dict(stream_id=sid,torch_peak_allocated=torch.cuda.max_memory_allocated(),torch_peak_reserved=torch.cuda.max_memory_reserved(),
                nvidia_smi_process_memory_snapshot=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()))
    assert len(rows)==640 and calls==(640 if a.method=='fp' else 0)
    assert not any(v for k,v in guard.snapshot()['counts'].items() if k.startswith('denied_'))
    metrics={}
    for name,subset in [('all_updates',rows),('startup_first8',[r for r in rows if r['update']<=8]),('steady_after8',[r for r in rows if r['update']>8])]:
        values=np.array([r['seconds'] for r in subset]);metrics[name]=dict(frames=len(values),mean_ms=float(values.mean()*1000),p50_ms=float(np.median(values)*1000),p95_ms=float(np.quantile(values,.95)*1000))
    result=dict(completed=True,method=a.method,startup_iterations=a.startup_iterations,checkpoint_sha256=a.expected_sha,source_sha256=source_hash(),entrypoint_sha256=sha(Path(__file__)),
        initializers_sha256=sha(a.initializers),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],streams=[s['stream_id'] for s in streams],
        fp_calls=calls,metrics=metrics,memory=memory,access_audit=guard.snapshot(),
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),
        torch_version=torch.__version__,scope='Single process, batch one, 20 fixed object streams; cached CPU RGB and metric depth to committed original-mesh pose returned on CPU, CUDA synchronized. Includes preprocessing, rendering, transfer and temporal update; excludes IO, model/mesh setup and first-observation priming. First eight updates reported in all_updates and excluded in steady_after8.',
        memory_limitation='PyTorch peak does not account for all native rasterizer/CUDA allocations. nvidia-smi snapshots include the CUDA context but are sampled per stream, not guaranteed device-memory peaks.')
    (a.out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));result['frames_sha256']=sha(a.out/'frames.jsonl')
    (a.out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(metrics))


if __name__=='__main__':main()
