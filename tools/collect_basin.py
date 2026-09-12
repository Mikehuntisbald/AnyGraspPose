"""Generate FP basin labels using only official train sequences and frozen models."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch,yaml
from lip.models.tracker import Tracker
from lip.models.basin import candidate_poses
from lip.data.clips import ClipDataset
from lip.engine.runtime import batch_step
from lip.geometry.renderer import Renderer
from lip.integrations.frozen_fp import FrozenFoundationPose
from lip.evaluation.metrics import errors
from lip.data.basin_targets import candidate_outcomes,label_policy

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);a=p.parse_args()
spec=yaml.safe_load(Path(a.spec).read_text());c=yaml.safe_load(Path(spec['source_config']).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
root=os.environ['DEX_YCB_DIR'];torch.set_num_threads(2);torch.manual_seed(spec['seed']);np.random.seed(spec['seed'])
model=Tracker(False).cuda().eval();ck=torch.load(spec['source_checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.requires_grad_(False)
fp=FrozenFoundationPose(c['foundationpose_root'],root,torch.device('cuda',0),c['foundationpose_refiner_sha256']);renderer=Renderer('cuda')
ds=ClipDataset(root,'cache/dexycb_s0',length=c['clip_length'],seed=spec['seed'],steps=spec['frames']*10,augmentation=True)
keys=sorted({s['subject_id']+'/'+s['sequence_id'] for s in ds.streams},key=lambda k:hashlib.sha256((str(spec['seed'])+k).encode()).hexdigest())
n=len(keys);split={k:('train' if i<int(.8*n) else 'calibration' if i<int(.9*n) else 'test') for i,k in enumerate(keys)}
policy=label_policy(spec.get('target_metric','adds_over_d'),spec.get('target_tau',.1),spec.get('target_margin',.005))
manifest=dict(schema_version=2,label_policy=policy,spec=spec,spec_sha256=sha(a.spec),source_checkpoint_sha256=sha(spec['source_checkpoint']),source_checkpoint_step=ck['global_step'],source_config_sha256=sha(spec['source_config']),fp_refiner_sha256=fp.weight_sha256,physical_sequence_split=split,official_data_split='s0_train_only',label='ADD-S < 0.1 diameter after FP(iteration=2)',candidate_names=['original','+5deg+0.02d','-5deg-0.02d','+15deg','+0.05d','random_0to30deg_0to0.1d'],selection='same declared train object/sequence/camera-balanced mixture, unique target frames',model_frozen=True,fp_frozen=True)
mp=out/'manifest.json'
if mp.exists():assert json.loads(mp.read_text())==manifest
else:mp.write_text(json.dumps(manifest,indent=2))
seen=set();positive=0;total=0;started=time.time();owned=0
for idx in range(len(ds)):
 sample,_=ds.choose(idx);sid=ds.streams[sample['stream']]
 key=(sid['stream_id'],int(sample['start']+(c['clip_length']+2)*sample['stride']))
 if key in seen:continue
 seen.add(key);frame_no=len(seen)-1
 if frame_no>=spec['frames']:break
 if frame_no%a.world!=a.rank:continue
 owned+=1;dest=out/f'frame_{frame_no:05d}.npz'
 item=None
 if dest.exists():
  with np.load(dest) as z:positive+=int(z['labels'].sum());total+=len(z['labels'])
 else:
  item=ds[idx];assert key[1]==int(item['frames'][c['clip_length']+3])
  g=torch.Generator().manual_seed(spec['seed']+frame_no);mode=str(np.random.default_rng(spec['seed']+frame_no).choice(['noisy_gt','lip_only','lip_fp'],p=spec['history_probabilities']))
  def observe(u,inp,pred):
   if u!=3:return
   base=inp['T_base_centered'][0];d=float(item['mesh']['diameter']);poses,deltas=candidate_poses(base,pred['pose_centered'][0],d,g)
   after=[]
   for pose in poses:
    after.append(fp(pose.detach(),item['rgb'][c['clip_length']+u-1],item['depth'][c['clip_length']+u-1],item['k'],sid['mesh_path'],item['mesh']['center']).cpu().numpy())
   gt=item['poses'][c['clip_length']+u];errs=[errors(t,gt,item['mesh']['vertices'],d) for t in after]
   rich=candidate_outcomes(poses.cpu().numpy(),np.stack(after),gt.numpy(),item['mesh']['vertices'],item['mesh']['center'],d,after_metrics=errs,metric=policy['primary_error'],tau=policy['tau'],margin=policy['improve_margin'])
   labels=rich['y_converge'].astype('f4')
   group=sid['subject_id']+'/'+sid['sequence_id'];meta=dict(index=frame_no,dataset_index=idx,stream_id=sid['stream_id'],frame_index=key[1],object_id=sid['object_id'],physical_sequence=group,split=split[group],history_mode=mode,post_fp_errors=errs)
   tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,z=pred['latent'][0].float().cpu().numpy(),delta=deltas.cpu().numpy(),labels=labels,candidate_poses=poses.cpu().numpy(),after_fp_poses=np.stack(after),base=base.cpu().numpy(),gt=gt.numpy(),meta=json.dumps(meta),**rich);tmp.replace(dest)
  with torch.no_grad():batch_step(model,[item],renderer,c,4,True,False,history_mode=mode,fp_transition=fp,observer=observe)
  with np.load(dest) as z:positive+=int(z['labels'].sum());total+=len(z['labels'])
 if owned%16==0:
  status=dict(rank=a.rank,owned_frames=owned,target=spec['frames'],candidates=total,positives=positive,seconds=time.time()-started);(out/f'status_rank{a.rank}.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
assert owned==len(range(a.rank,spec['frames'],a.world))
(out/f'complete_rank{a.rank}.json').write_text(json.dumps(dict(rank=a.rank,frames=owned,candidates=total,positives=positive,seconds=time.time()-started),indent=2))
