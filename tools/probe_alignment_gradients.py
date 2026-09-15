"""Actual final-checkpoint gradients, input ablations, and optimizer audit; no optimization."""
import json,sys,math
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import load_init,sha,source_hash
from lip.engine.config import check_data_gate
from lip.engine.stream_training import StreamTrainingModule,supervision_positions
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import angle
from lip.losses import pose_loss


def category(n):
 if n.startswith('rotation_alignment.'):
  return 'alignment_'+n.split('.')[1]
 if n.startswith('head.'):return 'parent_head'
 if n.startswith(('rgb_proj.','geometry.','fusion.')):return 'parent_visual_geometry'
 return 'parent_state_temporal'


def main():
 root=Path(__file__).resolve().parents[1];r=root/'runs/train_diagnostic';spec=json.loads((r/'spec.json').read_text());out=r/'gradient';out.mkdir(exist_ok=False);torch.set_num_threads(2)
 assert sha(spec['final_checkpoint'])==spec['final_checkpoint_sha256'] and source_hash()==spec['source_sha256']
 c=load_stream_config(spec['configuration']);audit=check_data_gate(spec['index_root'])
 model=make_model(c).cuda().eval();saved=load_init(spec['final_checkpoint'],model,audit,c);model.rgb.requires_grad_(False)
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 manifest=json.loads(Path(spec['training_manifest']).read_text());data=StreamClips(spec['data_root'],spec['index_root'],8,48,fixed=manifest,decode_threads=2,
  external_initializers=spec['train_initializers'],external_initializers_sha256=spec['train_initializers_sha256'],real_initialization_probability=.5,include_initial_observation=True)
 samples=[data[v['manifest_index']] for v in spec['cohort']['fit']];assert all(s['real_initialization_requested'] and 'real_initial_pose' in s for s in samples)
 capture=[];calls=[0]
 def hook(module,args,result):
  i=calls[0];calls[0]+=1
  if i in (1,4,8):capture.append(dict(update=i,inputs=tuple(v.detach().clone() for v in args),output=result.detach().clone()))
 h=model.rotation_alignment.register_forward_hook(hook)
 result=StreamTrainingModule(model,c,Renderer('cuda'))(samples,True);h.remove();pred=result['predictions']
 targets=torch.stack([s['targets'] for s in samples]).cuda().float();points=torch.stack([torch.as_tensor(s['mesh']['points']) for s in samples]).cuda().float();diam=torch.tensor([float(s['mesh']['diameter']) for s in samples],device='cuda')
 selected=torch.tensor(supervision_positions(8,48,8),device='cuda');parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad];groups={}
 for i,(n,p) in enumerate(parameters):groups.setdefault(category(n),[]).append(i)
 summaries={}
 for pop,mask in [('startup',selected&(torch.arange(56,device='cuda')<8)),('later',selected&(torch.arange(56,device='cuda')>=8))]:
  number=int(mask.sum());p=pred[:,mask].reshape(-1,4,4);target=targets[:,mask].reshape(-1,4,4)
  loss,detail=pose_loss(p,target,points.repeat_interleave(number,0),diam.repeat_interleave(number,0))
  losses={'translation':detail['translation'],'rotation_weighted':.5*detail['rotation'],'points':detail['points']};grads={};values={k:float(v.detach()) for k,v in losses.items()}
  for kind,value in losses.items():
   raw=torch.autograd.grad(value,[p for n,p in parameters],retain_graph=True,allow_unused=True)
   grads[kind]={g:torch.cat([(torch.zeros_like(parameters[i][1]) if raw[i] is None else raw[i]).float().flatten() for i in ids]) for g,ids in groups.items()}
  gs={}
  for g,ids in groups.items():
   rot=grads['rotation_weighted'][g];pts=grads['points'][g];trans=grads['translation'][g];total=rot+pts+trans
   def cosine(a,b):
    denominator=a.norm()*b.norm();return float((a@b)/denominator) if float(denominator)>1e-30 else None
   gs[g]=dict(parameter_numel=total.numel(),gradient_l2={k:float(v[g].norm()) for k,v in grads.items()},total_l2=float(total.norm()),total_rms=float(total.norm()/math.sqrt(total.numel())),
    rotation_point_cosine=cosine(rot,pts),rotation_total_cosine=cosine(rot,total),finite=bool(torch.isfinite(total).all()),nonzero=int(total.count_nonzero()))
   assert gs[g]['finite']
  summaries[pop]=dict(targets=number*len(samples),loss=values,groups=gs)
 del result,pred,targets,grads,losses,loss,detail
 torch.cuda.empty_cache()
 ablations=[];jacobians=[]
 for cap in capture:
  dense,geometry,latent=cap['inputs'];module=model.rotation_alignment
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   original=module(dense,geometry,latent);assert torch.equal(original,cap['output'])
   variants={'original':(dense,geometry,latent),'zero_dense':(torch.zeros_like(dense),geometry,latent),'zero_latent':(dense,geometry,torch.zeros_like(latent)),
    'swap_latent':(dense,geometry,latent.roll(1,0))}
   z=geometry.clone();z[:,2:7]=0;variants['zero_new_cad']=(dense,z,latent)
   z=geometry.clone();z[:,2:7]=z[:,2:7].roll(1,0);variants['swap_new_cad']=(dense,z,latent)
   for name,args in variants.items():
    value=module(*args).float();diff=(value-original.float()).norm(dim=-1)*180/torch.pi
    ablations.append(dict(update=cap['update'],variant=name,mean_output_deg=float(value.norm(dim=-1).mean()*180/torch.pi),mean_change_deg=float(diff.mean()),max_change_deg=float(diff.max()),per_clip_change_deg=diff.cpu().tolist()))
  inputs=tuple(v.detach().float().requires_grad_(True) for v in cap['inputs']);value=module(*inputs)
  squared=[torch.zeros(len(samples),device='cuda') for _ in inputs]
  for axis in range(3):
   grads=torch.autograd.grad(value[:,axis].sum(),inputs,retain_graph=axis<2)
   for i,g in enumerate(grads):squared[i]+=g.detach().flatten(1).square().sum(1)
  jacobians.append(dict(update=cap['update'],precision='FP32 branch on frozen BF16-trajectory inputs',input_jacobian_frobenius={name:q.sqrt().cpu().tolist() for name,q in zip(('fused_dense','geometry','parent_latent'),squared)}))
 assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items());assert sha(spec['final_checkpoint'])==spec['final_checkpoint_sha256']
 optimizer=[]
 for group in saved['optimizer']['param_groups']:
  states=[saved['optimizer']['state'][i] for i in group['params'] if i in saved['optimizer']['state']]
  optimizer.append(dict(category=group['category'],lr=group['lr'],parameters=len(group['params']),state_entries=len(states),steps=sorted({int(s['step']) for s in states}),moment_norm_sum=sum(float(s['exp_avg'].norm()) for s in states)))
 receipt=dict(completed=True,checkpoint_sha256=spec['final_checkpoint_sha256'],spec_sha256=sha(r/'spec.json'),source_sha256=source_hash(),entrypoint_sha256=sha(Path(__file__)),
  samples=spec['cohort']['fit'],gradients=summaries,input_ablations=ablations,jacobians=jacobians,optimizer=optimizer,weights_bitwise_unchanged=True,optimizer_steps=0,
  scope='Eight angle-stratified real train clips; same recorded final weights. Full 56-update training graph, original losses/occluders. Input ablations hold other branch inputs fixed and are off-manifold sensitivity probes, not trajectory accuracy or proof of information absence. Parent visual features already contain geometry, so zero_new_cad only removes the additional CAD route.')
 (out/'report.json').write_text(json.dumps(receipt,indent=2,allow_nan=False));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
