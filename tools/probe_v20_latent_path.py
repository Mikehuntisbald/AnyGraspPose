"""Frozen paired pose-response and geometry-readout interventions on fixed40 val sequences.
GT pose bases/surfaces/support are diagnostic oracles, never deployable scores.
"""
import argparse,json,sys,time,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np,torch,yaml,cv2
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.features import prepare_scene,encode_scenes
from lip.unified.renderer import FullTextureRenderer
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import center_pose,update,log,angle
from lip.models.stream_tracker import StreamTracker
from lip.engine.stream_features import build_current_features,stack_current
from lip.engine.stream_state import FrameMeta
from lip.engine.jepa_checkpoint import load_core,sha
from lip.evaluation.metrics import errors

ROOT=Path('/mnt/why/dexycb_lip')
LIP=ROOT/'smooth_val_evaluation_20260915/runs/full'
V20=ROOT/'unified_jepa_20260921/pose_geometry_v20'


def digest(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def fit_rigid(x,y):
 xc=x-x.mean(0);yc=y-y.mean(0);u,s,vh=torch.linalg.svd(xc.T@yc)
 fix=torch.eye(3,device=x.device);fix[-1,-1]=torch.det(vh.T@u.T)
 r=vh.T@fix@u.T;t=y.mean(0)-r@x.mean(0);pose=torch.eye(4,device=x.device);pose[:3,:3]=r;pose[:3,3]=t
 return pose,s


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,default=8);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 outdir=Path(a.out);outdir.mkdir(parents=True,exist_ok=False);torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(42);cv2.setNumThreads(0)
 c=yaml.safe_load(Path(a.config).read_text());c['runtime']['compile_frame']=False;c['runtime']['compile_loss']=False
 m=build_model(c);checkpoint=V20/'runs/seed42/last.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['step']==45400;load_core(m,saved['model']);del saved
 m.requires_grad_(False).eval();m.weights_version=sha(checkpoint)
 old=torch.load(LIP/'selected.pt',map_location='cpu',weights_only=False);lc=old['config']
 lip=StreamTracker('stream_dual_cross_residual',lc['memory_frames'],lc.get('dropout',0.),False,lc.get('time_unit',1/30),lc.get('max_gap_seconds',.5),'functional').cuda();lip.load_state_dict(old['model']);lip.requires_grad_(False).eval();del old
 before={k:digest(v) for k,v in [('v20',m),('lip',lip)]}
 captured=[];latent_rows=[];latent_ids=[]
 m.core.blocks[0].register_forward_pre_hook(lambda _module,args:captured.append(args[0]))
 def pooled(x):return F.avg_pool2d(x.float().transpose(1,2).reshape(1,256,16,16),4).flatten().cpu().half()
 store=make_store(c,m);renderer=FullTextureRenderer('cuda');geo_renderer=Renderer('cuda')
 index=Path(c['paths']['index_root']);root=Path(c['paths']['data_root']);audit=json.loads((index/'audit.json').read_text());fps=audit['fps']
 registry={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines()) if s['split']=='val'}
 initial=json.loads(Path(c['paths']['val_initializers']).read_text())['initializers']
 refs={name:{(r['stream_id'],r['frame_index']):r for r in map(json.loads,path.read_text().splitlines())} for name,path in [('lip',LIP/'residual/scored/predictions.jsonl'),('v20',V20/'validation/step45400_scored/predictions.jsonl')]}
 metadata=json.loads(Path(c['cad_surface']['models_info']).read_text());chosen={}
 for sid,s in sorted(registry.items()):
  if initial[sid] is not None and s['num_frames']-initial[sid]['frame_index']>=40:chosen.setdefault('/'.join(sid.split('/')[:2]),sid)
 assert len(chosen)==40
 selected=sorted(chosen.items())[a.rank::a.world]
 if a.smoke:selected=selected[:1]
 started=time.monotonic();rows=0;replay=[];selection=[];alignment=[]
 def meta(f,first,diag):return FrameMeta(torch.tensor([f/fps],device='cuda',dtype=torch.float64),torch.tensor([f-first+1],device='cuda'),torch.zeros(1,device='cuda',dtype=torch.long),torch.ones(1,17,device='cuda',dtype=torch.bool),diag['role_bias'][None])
 with torch.no_grad(),(outdir/'frames.jsonl').open('w') as writer:
  for physical,sid in selected:
   s=registry[sid];first=initial[sid]['frame_index'];directory=root/s['relative_dir']
   with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
   center=torch.tensor(mesh['center'],device='cuda');k=torch.tensor(s['intrinsics'],device='cuda');d=float(mesh['diameter']);points=mesh['points'];cad=store.get(root/s['mesh_path'],mesh)
   with np.load(index/s['pose_cache']) as z:gt=center_pose(torch.tensor(z['poses'],device='cuda'),center)
   pool=list(range(first+8,s['num_frames']));worst=min(pool,key=lambda f:2. if refs['lip'][sid,f]['visibility'] is None else refs['lip'][sid,f]['visibility'])
   frames=sorted(set([first+8,min(first+24,s['num_frames']-1),worst]))
   if a.smoke:frames=frames[:1]
   selection.append(dict(stream_id=sid,physical_sequence=physical,frames=frames,object_id=s['object_id']))
   cache=None
   for f in range(first,max(frames)+1):
    im=cv2.imread(str(directory/f'color_{f:06d}.jpg'));dep=cv2.imread(str(directory/f'aligned_depth_to_color_{f:06d}.png'),-1)
    assert im is not None and dep is not None
    rgb=torch.tensor(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy(),device='cuda').float()/255
    depth=torch.tensor(dep.astype('f4')[None],device='cuda')*audit['depth_scale_to_m']
    base=center_pose(torch.tensor(initial[sid]['pose_original'],device='cuda'),center) if f==first else torch.tensor(refs['lip'][sid,f-1]['pose_centered'],device='cuda')
    previous=None if f<=first+1 else torch.tensor(refs['lip'][sid,f-2]['pose_centered'],device='cuda')
    features,diag=build_current_features(rgb,depth,base,k,mesh,geo_renderer,f/fps,(f-1)/fps,previous,None if previous is None else (f-2)/fps)
    old_meta=meta(f,first,diag);incoming=cache
    with torch.autocast('cuda',dtype=torch.bfloat16):old_out,cache=lip(stack_current([features]),old_meta,incoming)
    if f>first:replay.append(float((old_out['pose_centered'][0]-torch.tensor(refs['lip'][sid,f]['pose_centered'],device='cuda')).abs().max()))
    if f not in frames:continue
    symmetry=bool(metadata[str(s['object_id'])].get('symmetries_discrete') or metadata[str(s['object_id'])].get('symmetries_continuous'))
    bases=[('lip_previous',base,previous),('v20_previous',torch.tensor(refs['v20'][sid,f-1]['pose_centered'],device='cuda'),torch.tensor(refs['v20'][sid,f-2]['pose_centered'],device='cuda')),('gt_zero',gt[f],None)]
    for axis in range(6):
     for sign in (-1,1):
      perturb=torch.zeros(1,6,device='cuda');perturb[0,axis]=sign*(torch.pi/18 if axis<3 else .05)
      bases.append((f'axis{axis}_{sign:+d}',update(gt[f:f+1],perturb[:,:3],perturb[:,3:],gt.new_tensor([d]))[0],None))
    for tag,base,previous in bases:
     scene=prepare_scene(rgb,depth,base,mesh,k,f/fps,sid,cad,renderer,previous,(f-1)/fps)
     obs=encode_scenes(m,[scene],frame_id=f-first)
     captured.clear()
     with torch.autocast('cuda',dtype=torch.bfloat16):result,_=m(obs,None,torch.zeros(1,device='cuda',dtype=torch.bool))
     feats,diagnostic=build_current_features(rgb,depth,base,k,mesh,geo_renderer,f/fps,(f-1)/fps,previous,None if previous is None else (f-2)/fps)
     alignment.append(float((scene.affine-diagnostic['A']).abs().max()))
     shared=stack_current([feats]);current_meta=meta(f,first,diagnostic)
     with torch.autocast('cuda',dtype=torch.bfloat16):
      cold,_=lip(shared,current_meta,None);warm,_=lip(shared,current_meta,incoming)
     poses={'base':base,'v20':result['pose_centered'][0],'lip_no_history':cold['pose_centered'][0],'lip_history':warm['pose_centered'][0]}
     if tag=='gt_zero' or tag.startswith('axis'):
      assert len(captured)==1
      latent_rows.append(dict(fused=pooled(captured[0]),patch=pooled(result['patch_latent']),object=result['latent'].float().flatten().cpu().half(),lip_object=cold['latent'].float().flatten().cpu().half()))
      latent_ids.append(dict(physical_sequence=physical,stream_id=sid,frame=f,tag=tag,symmetry=symmetry))
     delta=lambda o:torch.cat((o['delta_rotvec'][0],o['delta_center_norm'][0])).float().cpu().tolist()
     deltas={'v20':delta(result),'lip_no_history':delta(cold),'lip_history':delta(warm)}
     if tag in ('lip_previous','v20_previous','gt_zero','axis0_+1','axis1_+1','axis2_+1'):
      surf=torch.cat((result['surface_xyz'],result['surface_depth_residual'],result['geometry_valid_logits']),1)
      valid=F.avg_pool2d(obs.packet.pixel_valid.float(),14,14).flatten(1)>=.999
      def read(surface,relation=True):
       with torch.autocast('cuda',dtype=torch.bfloat16):
        if relation:q,_=m.read_object(result['patch_latent'],valid,surface,obs.geometry_image,obs.base,obs.cad_valid,result['evidence_logits'],(obs.object_xyz,obs.depth_valid))
        else:
         q=m.query.expand(1,-1,-1);q=q+m.object_attn(m.object_norm(q),result['patch_latent'],valid)
        latent=F.layer_norm(q[:,0],(256,));change=m.head(latent).float()
       return update(base[None],change[:,:3],change[:,3:],obs.diameter)[0]
      repeated=read(surf);assert torch.allclose(repeated,result['pose_centered'][0],atol=2e-5,rtol=1e-4)
      off=surf.clone();off[:,4]=-30;poses['completion_off']=read(off);poses['relation_off']=read(surf,False)
      render=renderer(cad['appearance'],gt[f],scene.k_crop,224);mask=render['mask'][None,None]
      truth_xyz=render['xyz'][None]/d;truth_depth=(render['depth'][None]-base[2,3])/d
      oracle=surf.clone();oracle[:,:3]=torch.where(mask,truth_xyz,oracle[:,:3]);oracle[:,3:4]=torch.where(mask,truth_depth,oracle[:,3:4]);poses['oracle_surface_values']=read(oracle)
      oracle[:,4:5]=torch.where(mask,12.,-12.);poses['oracle_surface_and_validity']=read(oracle)
      # Diagnostic rigid fit: GT silhouette identifies support, but values are predicted.
      yy,xx=torch.meshgrid(torch.arange(224,device='cuda'),torch.arange(224,device='cuda'),indexing='ij')
      rays=torch.stack((xx,yy,torch.ones_like(xx)),-1).float()@torch.linalg.inv(scene.k_crop).T
      predicted_camera=rays*result['surface_depth_m'][0,0,:,:,None]
      x=result['surface_xyz'][0].permute(1,2,0)*d;y=predicted_camera
      support=(-F.max_pool2d(-mask.float(),5,1,2)>.999)[0,0];support=support&torch.isfinite(x).all(-1)&torch.isfinite(y).all(-1)&(y[:,:,2]>.001)
      grid=torch.zeros_like(support);grid[::4,::4]=True;support=support&grid
      if int(support.sum())>=6:
       fitted,singular=fit_rigid(x[support],y[support]);poses['predicted_geometry_fit_gt_support']=fitted
     stats={name:errors(pose.cpu().numpy(),gt[f].cpu().numpy(),points,d,dtype='f4') for name,pose in poses.items()}
     effects={n:dict(rotation_deg=float(angle(p[:3,:3]@result['pose_centered'][0,:3,:3].T))*180/torch.pi,center_mm=float((p[:3,3]-result['pose_centered'][0,:3,3]).norm())*1000) for n,p in poses.items() if n not in ('base','v20','lip_no_history','lip_history')}
     row=dict(effects=effects,physical_sequence=physical,stream_id=sid,frame_index=f,object_id=s['object_id'],symmetry=symmetry,visibility=refs['lip'][sid,f]['visibility'],base_kind=tag,metrics=stats,deltas=deltas,
      relation_gain=float(m.geometry_readout.gain.sigmoid()),completed_weight=float(result['relation_completed_weight'].mean()),measured_weight=float(result['relation_measured_weight'].mean()),
      native_pose_max_abs=float((result['pose_centered'][0]-torch.tensor(refs['v20'][sid,f]['pose_centered'],device='cuda')).abs().max()) if tag=='v20_previous' else None)
     writer.write(json.dumps(row,allow_nan=False)+'\n');rows+=1
    writer.flush()
   print(json.dumps(dict(rank=a.rank,stream=sid,rows=rows,seconds=time.monotonic()-started)),flush=True)
 np.savez_compressed(outdir/'latents.npz',**{k:torch.stack([r[k] for r in latent_rows]).numpy() for k in latent_rows[0]})
 (outdir/'latent_ids.json').write_text(json.dumps(latent_ids))
 after={k:digest(v) for k,v in [('v20',m),('lip',lip)]};assert before==after
 receipt=dict(completed=True,training=False,optimizer_updates=0,official_test_access=False,rank=a.rank,world=a.world,smoke=a.smoke,rows=rows,selection=selection,checkpoint_sha256=m.weights_version,lip_checkpoint_sha256=sha(LIP/'selected.pt'),state_digests_before=before,state_digests_after=after,
  maximum_crop_difference=max(alignment),old_lip_replay_max_abs=max(replay),seconds=time.monotonic()-started,frames_sha256=sha(outdir/'frames.jsonl'),entrypoint_sha256=sha(__file__),scope='Fixed40 physical sequences; up to3 selected frames each; conditional perturbation diagnostics. GT bases/surface/support are oracles; point scores use cached mesh points. Controlled pose cases have no motion-state hint; LIP history is causal original-only replay.')
 (outdir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
