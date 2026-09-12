"""Add FP basin-boundary probes; fresh sequence groups supply a new validation set."""
import argparse,json,hashlib,time,os
from pathlib import Path
import numpy as np,torch,yaml
from lip.data.clips import ClipDataset
from lip.data.basin_targets import candidate_outcomes,label_policy
from lip.models.tracker import Tracker
from lip.models.basin import candidate_poses
from lip.engine.runtime import batch_step
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import exp,log
from lip.integrations.frozen_fp import FrozenFoundationPose
from lip.evaluation.metrics import errors
p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,default=4);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(2);torch.manual_seed(1731)
source=Path('runs/basin_v1/outcomes_v2');manifest=json.loads((source/'manifest.json').read_text());s=manifest['spec'];c=yaml.safe_load(Path(s['source_config']).read_text());root=os.environ['DEX_YCB_DIR']
with np.load(source/'outcomes.npz') as z:table={k:z[k].copy() for k in z.files}
metas=[json.loads(str(x)) for x in table['meta']];used={m['physical_sequence'] for m in metas};allgroups=set(manifest['physical_sequence_split']);fresh=sorted(allgroups-used);assert len(fresh)>=40
plan=[]
for i,m in enumerate(metas):
 if m['split']!='test':plan.append(dict(old_index=i,dataset_index=m['dataset_index'],split=m['split'],history_mode=m['history_mode']))
ds=ClipDataset(root,'cache/dexycb_s0',length=c['clip_length'],seed=s['seed'],steps=1_000_000,augmentation=True)
seen=set();test_count=0
for ix in range(max(m['dataset_index'] for m in metas)+1,len(ds)):
 x,_=ds.choose(ix);stream=ds.streams[x['stream']];group=stream['subject_id']+'/'+stream['sequence_id'];key=(stream['stream_id'],x['start']+(c['clip_length']+2)*x['stride'])
 if group not in fresh or key in seen:continue
 seen.add(key);mode=str(np.random.default_rng(1731+test_count).choice(['noisy_gt','lip_only','lip_fp'],p=[.4,.3,.3]));plan.append(dict(old_index=None,dataset_index=ix,split='test',history_mode=mode));test_count+=1
 if test_count==320:break
assert test_count==320
run=dict(source_data_sha256=json.loads((source/'complete.json').read_text())['data_sha256'],spec=s,source_checkpoint=s['source_checkpoint'],fresh_groups=fresh,plan=plan,label_policy=label_policy(),candidate_protocol='original six, far probe (30-90deg and 0.1-0.3d), three bisections when endpoint convergence differs; otherwise seven valid candidates',max_candidates=10,validation='320 unique frames from physical sequences absent from all v1/v2 critic data')
if a.rank==0:(out/'manifest.json').write_text(json.dumps(run,indent=2))
actor=Tracker(False).cuda().eval().requires_grad_(False);actor.load_state_dict(torch.load(s['source_checkpoint'],map_location='cpu',weights_only=False)['model']);fp=FrozenFoundationPose(c['foundationpose_root'],root,torch.device('cuda',0),c['foundationpose_refiner_sha256']);renderer=Renderer('cuda');started=time.time();payloads=[];frame_ids=[]
for fi,entry in enumerate(plan):
 if fi%a.world!=a.rank:continue
 item=ds[entry['dataset_index']];stream=item['stream'];d=float(item['mesh']['diameter']);u=3;at=c['clip_length']+u-1;gt=item['poses'][c['clip_length']+u].numpy()
 if entry['old_index'] is not None:
  oi=entry['old_index'];z=table['z'][oi];base=torch.from_numpy(table['base'][oi]).cuda();poses=[torch.from_numpy(x).cuda() for x in table['candidate_pose'][oi]];after=[x.copy() for x in table['refined_pose'][oi]]
 else:
  captured={}
  def observe(step,inp,pred):
   if step==3:captured.update(z=pred['latent'][0].float().cpu().numpy(),base=inp['T_base_centered'][0].clone(),pred=pred['pose_centered'][0].clone())
  with torch.no_grad():batch_step(actor,[item],renderer,c,4,True,False,history_mode=entry['history_mode'],fp_transition=fp,observer=observe)
  z=captured['z'];base=captured['base'];pp,_=candidate_poses(base,captured['pred'],d,torch.Generator().manual_seed(1731+fi));poses=list(pp.unbind());after=[fp(pose.detach(),item['rgb'][at],item['depth'][at],item['k'],stream['mesh_path'],item['mesh']['center']).cpu().numpy() for pose in poses]
 g=torch.Generator().manual_seed(913731+fi)
 def axis():
  x=torch.randn(3,generator=g).cuda();return x/x.norm().clamp_min(1e-8)
 rv=axis()*((30+60*float(torch.rand((),generator=g)))*torch.pi/180);tv=axis()*(.1+.2*float(torch.rand((),generator=g)));anchor=poses[0]
 def propose(alpha):
  t=anchor.clone();t[:3,:3]=exp(alpha*rv)@anchor[:3,:3];t[:3,3]+=d*alpha*tv;return t
 def add(alpha):
  t=propose(alpha);refined=fp(t.detach(),item['rgb'][at],item['depth'][at],item['k'],stream['mesh_path'],item['mesh']['center']).cpu().numpy();poses.append(t);after.append(refined);return errors(refined,gt,item['mesh']['vertices'],d)['adds_01']
 first=errors(after[0],gt,item['mesh']['vertices'],d)['adds_01'];last=add(1.);alphas=[None]*6+[1.]
 if first!=last:
  lo,hi=0.,1.
  for _ in range(3):
   mid=(lo+hi)/2;ym=add(mid);alphas.append(mid)
   if ym==first:lo=mid
   else:hi=mid
 candidates=torch.stack(poses);deltas=torch.cat((log(candidates[:,:3,:3]@base[:3,:3].T),(candidates[:,:3,3]-base[:3,3])/d),-1)
 rich=candidate_outcomes(candidates.cpu().numpy(),np.stack(after),gt,item['mesh']['vertices'],item['mesh']['center'],d)
 n=len(poses);payload={}
 for key,val in rich.items():
  if np.asarray(val).ndim and len(val)==n and key not in ['mesh_center_m']:
   pad=np.zeros((10,)+val.shape[1:],dtype=val.dtype);pad[:n]=val;payload[key]=pad
  else:payload[key]=val
 dd=np.zeros((10,6),dtype='f4');dd[:n]=deltas.cpu().numpy();group=stream['subject_id']+'/'+stream['sequence_id']
 meta=dict(index=fi,old_index=entry['old_index'],dataset_index=entry['dataset_index'],stream_id=stream['stream_id'],frame_index=int(item['frames'][c['clip_length']+3]),object_id=stream['object_id'],physical_sequence=group,split=entry['split'],history_mode=entry['history_mode'],probe_alphas=alphas)
 payload.update(z=z,delta=dd,valid=np.arange(10)<n,base=base.cpu().numpy(),gt=gt,meta=np.array(json.dumps(meta)));payloads.append(payload);frame_ids.append(fi)
 if len(payloads)%16==0:print(json.dumps(dict(rank=a.rank,frames=len(payloads),total_owned=len(range(a.rank,len(plan),a.world)),seconds=time.time()-started)),flush=True)
tmp=out/f'rank{a.rank}.tmp.npz';np.savez_compressed(tmp,frame_ids=np.array(frame_ids),**{k:np.stack([p[k] for p in payloads]) for k in payloads[0]});tmp.replace(out/f'rank{a.rank}.npz')
(out/f'rank{a.rank}.complete.json').write_text(json.dumps(dict(frames=len(payloads),seconds=time.time()-started)))
