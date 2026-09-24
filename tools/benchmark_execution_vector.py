"""Matched eight-H20 zero-update speed/parity gate for execution-only changes."""
import argparse,json,os,sys,time,copy
from pathlib import Path
import torch,torch.distributed as dist,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.reconstruction_only import initialize_training
from lip.unified.optimized_training import OptimizedEpisode,synchronize_gradients
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,build_teachers
from lip.unified.execution_speed import FrozenFeatureGraph
from lip.engine.jepa_checkpoint import load_core,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 rank=int(os.environ['RANK']);torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(42)
 torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False;dist.init_process_group('nccl')
 c=yaml.safe_load(Path(a.config).read_text());model=build_model(c);source,_,reference=initialize_training(model,c,8);del source
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);load_core(model,saved['model']);del saved
 factory=Factory(c,model,make_store(c,model))
 inputs,targets=zip(*(factory.sample(42000139+rank*4+i,frames=40) for i in range(4)))
 # Render/target parity is checked separately from end-to-end gradient parity.
 scenes=[];truth=[];visible=[];masks=[]
 for e,(gt,vis) in zip(inputs,targets):
  for frame in range(3):
   scenes.append(prepare_scene(e.rgb[frame],e.depth[frame],e.initial,e.mesh,e.k,e.times[frame],e.stream,e.cad,factory.renderer))
   truth.append(gt[frame]);visible.append(vis[frame]);masks.append(torch.ones_like(scenes[-1].bounds))
 teacher_args=(model.ema_teacher,scenes,torch.stack(truth),torch.stack(visible),masks,factory.renderer)
 ta=build_teachers(*teacher_args,real_geometry_max_radius_d=1.)
 tb=build_teachers(*teacher_args,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True)
 target_errors={}
 for name in ta.__dataclass_fields__:
  x,y=getattr(ta,name),getattr(tb,name)
  if not isinstance(x,torch.Tensor):continue
  if x.dtype==torch.bool:assert torch.equal(x,y),name
  else:
   assert torch.allclose(x,y,rtol=1e-6,atol=2e-6,equal_nan=True),name
   target_errors[name]=float((x-y).abs().nan_to_num().max())
 # Graphs must follow in-place EMA writes and must not return aliased outputs.
 graph_checks=[]
 for encoder in (reference.encoder,model.ema_teacher):
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   image=torch.randn(8,3,224,224,device='cuda');n=[v-1 for v in encoder.feature_layers]
   graph=FrozenFeatureGraph(encoder)
   first=graph(image,n,False,True)
   expected=encoder.backbone.get_intermediate_layers(image,n=n,reshape=False,norm=True)
   assert all(torch.equal(x,y) for x,y in zip(first,expected))
   bias=encoder.backbone.patch_embed.proj.bias;backup=bias.clone();bias.add_(.001*torch.arange(len(bias),device='cuda'))
   changed=graph(image,n,False,True);new_expected=encoder.backbone.get_intermediate_layers(image,n=n,reshape=False,norm=True)
   assert all(torch.equal(x,y) for x,y in zip(changed,new_expected))
   assert any(not torch.equal(x,y) for x,y in zip(first,changed))
   assert all(torch.equal(x,y) for x,y in zip(first,expected))
   bias.copy_(backup);graph_checks.append(True)
   del graph
 arms=[('baseline',{}),
       ('vectorized',dict(reference_dino_graph=True,teacher_dino_graph=True,fast_geometry=True,batch_teacher_render=True,vectorized_teacher=True)),
       ('vector_geometry',dict(reference_dino_graph=True,teacher_dino_graph=True,fast_geometry=True,batch_teacher_render=True,vectorized_teacher=True,vector_geometry=True))]
 results={};baseline=None;baseline_loss=None;poses=None
 for name,flags in arms:
  current=copy.deepcopy(c);current['runtime'].update(flags)
  run=OptimizedEpisode(model,factory.renderer,current,reference_model=reference)
  times=[];losses=[];components=[]
  for repeat in range(4):
   model.zero_grad(set_to_none=True);dist.barrier();torch.cuda.synchronize();tick=time.monotonic()
   loss,stats=run(inputs,targets);synchronize_gradients(model.parameters());torch.cuda.synchronize()
   elapsed=torch.tensor(time.monotonic()-tick,device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
   times.append(float(elapsed));losses.append(float(loss));components.append(run.profile.seconds())
   if repeat==3:
    grad=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    assert grad.isfinite().all()
    if baseline is None:baseline=grad.clone();baseline_loss=float(loss);poses=torch.stack(run.last_poses).clone()
    cosine=float(torch.nn.functional.cosine_similarity(grad,baseline,dim=0));relative=float((grad-baseline).norm()/baseline.norm())
    exact_crop=torch.equal(poses,torch.stack(run.last_poses));assert exact_crop,(name,'crop changed')
    assert abs(float(loss)-baseline_loss)<max(1e-5,abs(baseline_loss)*.002),(name,'loss changed')
    assert cosine>.9999 and relative<.01,(name,cosine,relative)
  item=dict(seconds=times,warm_seconds=sum(times[1:])/3,losses=losses,flags=flags,gradient_cosine=cosine,gradient_relative_error=relative,crop_exact=exact_crop,components=components[-1],peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
  results[name]=item
  if rank==0:print(json.dumps(dict(arm=name,result=item)),flush=True)
  del run
 best=min((k for k in results if k!='baseline'),key=lambda k:results[k]['warm_seconds'])
 speedup=results['baseline']['warm_seconds']/results[best]['warm_seconds']
 receipt=dict(passed=True,checkpoint_sha256=sha(a.checkpoint),persisted_updates=0,world=8,arms=results,best=best,speedup=speedup,target_max_errors=target_errors,graph_ema_update_checks=graph_checks)
 if rank==0:Path(a.out).write_text(json.dumps(receipt,indent=2))
 dist.destroy_process_group()

if __name__=='__main__':main()
