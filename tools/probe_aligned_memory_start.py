"""Real-fragment zero-residual and pose-loss gradient check for a prototype."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.models.aligned_memory import AlignedMemoryReadout,pool_memory_geometry
from lip.models.stream_rk import AnchorBank
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_training import StreamTrainingModule
from lip.engine.stream_checkpoint import sha,source_hash
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import update
from lip.losses import pose_loss


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','checkpoint','data-root','index-root','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--query-side',type=int,choices=(4,14),default=4)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    c=load_stream_config(a.config);assert c['architecture_id']=='stream_rk_spatial'
    model=make_model(c).cuda().eval();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    for parameter in model.parameters():parameter.requires_grad_(False)
    ds=StreamClips(a.data_root,a.index_root,8,48,seed=42);samples=[ds[i] for i in range(2)]
    meshes=[s['mesh'] for s in samples];points=torch.stack([torch.as_tensor(m['points'],device='cuda') for m in meshes]);diameter=torch.stack([torch.as_tensor(m['diameter'],device='cuda') for m in meshes])
    renderer=Renderer('cuda');encode=model.encode_dense;forward=model.forward;results={}
    for precision in ('fp32','bf16'):
        torch.manual_seed(42);readout=AlignedMemoryReadout().cuda();geometry_cache={};captured={};losses=[];checked=[];iteration=0;projection_checks=[]
        def capture(features):
            source,dense=encode(features);captured['source']=source.detach();captured['dense']=dense.detach();return source,dense
        def check_projection(geometry,features,side):
            with torch.autocast('cuda',enabled=False):
                base=features['T_base_centered'].float();points=geometry[...,:3]*features['object_diameter_m'][:,None,None]
                camera=points@base[:,:3,:3].transpose(-1,-2)+base[:,None,:3,3]
                intrinsics=features['state_input'][:,10:14].float()*224
                uv=camera[...,:2]/camera[...,2:].clamp_min(1e-6)*intrinsics[:,None,:2]+intrinsics[:,None,2:]
                cell=torch.arange(side*side,device='cuda');lower=torch.stack((cell%side,cell//side),-1).float()*224/side-.5;upper=lower+224/side
                overflow=torch.maximum(lower[None]-uv,uv-upper[None]).clamp_min(0).amax(-1)
                maximum=float(torch.where(geometry[...,4]>0,overflow,0.).max());assert maximum<.01,(side,maximum)
                projection_checks.append(maximum)
        def side_probe(features,meta,cache=None,profiler=None):
            nonlocal iteration
            out,next_cache=forward(features,meta,cache,profiler)
            b=len(features['rgb']);bank=getattr(cache,'anchors',None) or AnchorBank.empty(out['latent_object'],4)
            patches=getattr(cache,'patches',None)
            if patches is None:patches=out['latent_object'].new_zeros(b,4,196,256)
            current_geometry=pool_memory_geometry(features['geometry'],a.query_side)
            dense_geometry=pool_memory_geometry(features['geometry'],14)
            check_projection(current_geometry,features,a.query_side);check_projection(dense_geometry,features,14)
            stored=torch.stack([torch.stack([geometry_cache[(lane,int(frame))] if bool(valid) else torch.zeros_like(dense_geometry[lane]) for frame,valid in zip(bank.frame_id[lane],bank.valid[lane])]) for lane in range(b)])
            for lane in range(b):geometry_cache[(lane,int(meta.frame_id[lane]))]=dense_geometry[lane].detach()
            # The original predictor/state transaction is already complete.
            # GT below only supplies a loss for the external prototype.
            with torch.enable_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                queries=captured['source'][:,:16] if a.query_side==4 else captured['dense']
                residual,diag=readout(out['latent_object'].detach(),queries,current_geometry,patches.detach(),stored,bank.detach(),meta,
                    model.pose_condition(bank,features,meta).detach())
                delta=model.head(out['latent'].detach()+residual).float()
                delta=delta*(1-.5*torch.tanh(model.update_strength)*(1-out['observation_support'].detach()))[:,None]
                with torch.autocast('cuda',enabled=False):
                    candidate=update(features['T_base_centered'].float(),delta[:,:3],delta[:,3:],features['object_diameter_m'].float())
                assert torch.equal(candidate.detach(),out['pose_centered']),('Nonzero initial perturbation',precision,iteration)
                checked.append(dict(frame=iteration,usable_tokens=diag['usable_tokens'].detach().cpu().tolist(),usable_queries=diag['usable_queries'].detach().cpu().tolist()))
                if iteration>=8:
                    target=torch.stack([s['targets'][iteration].cuda() for s in samples]);loss,_=pose_loss(candidate,target,points,diameter);losses.append(loss)
            iteration+=1
            return out,next_cache
        model.encode_dense=capture;model.forward=side_probe
        with torch.no_grad():baseline=StreamTrainingModule(model,dict(c,precision=precision,temporal_occlusion_probability=0.),renderer)(samples,True)
        loss=torch.stack(losses).mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in readout.parameters())
        assert readout.attention.out_proj.weight.grad.abs().sum()>0
        assert all(p.grad is None for p in model.parameters())
        results[precision]=dict(frames=len(checked)*len(samples),zero_residual_pose_bitwise_equal=True,finite_true_pose_loss_gradient=True,
            mean_pose_loss=float(loss.detach()),out_projection_gradient_l1=float(readout.attention.out_proj.weight.grad.abs().sum()),
            max_usable_tokens=max(max(x['usable_tokens']) for x in checked),max_usable_queries=max(max(x['usable_queries']) for x in checked),parent_parameters_received_no_gradients=True,
            max_rendered_coordinate_projection_overflow_px=max(projection_checks))
        model.encode_dense=encode;model.forward=forward
        del readout,baseline,loss,losses;torch.cuda.empty_cache()
    report=dict(passed=True,scope='Two real official-train fragments, 8+48 frames each. External zero-initialized readout preserves frozen spatial model poses and receives actual GT pose-loss gradients. No optimizer step, model integration, formal training, validation selection, or FP calls.',
        prototype_source_sha256=source_hash(),query_side=a.query_side,frozen_checkpoint_sha256=sha(a.checkpoint),samples=[s['sample'] for s in samples],results=results)
    (a.out/'start.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
