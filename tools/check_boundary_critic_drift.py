"""Frozen critic check on all fresh sequence contexts under the resume actor encoder."""
import argparse,json,hashlib,os
from pathlib import Path
import numpy as np,torch,yaml
from sklearn.metrics import roc_auc_score
from lip.models.tracker import Tracker
from lip.models.basin import load_frozen_basin
from lip.data.clips import ClipDataset
from lip.engine.runtime import batch_step
from lip.geometry.renderer import Renderer
from lip.integrations.frozen_fp import FrozenFoundationPose
p=argparse.ArgumentParser();p.add_argument('--actor',required=True);p.add_argument('--out',required=True);a=p.parse_args()
j=Path('runs/basin_boundary_v3');m=json.loads((j/'data/manifest.json').read_text());s=m['spec'];c=yaml.safe_load(Path(s['source_config']).read_text());torch.set_num_threads(2);torch.manual_seed(731)
with np.load(j/'data/outcomes.npz') as f:table={k:f[k].copy() for k in f.files}
meta=[json.loads(str(x)) for x in table['meta']];source=Tracker(False).cuda().eval().requires_grad_(False);source.load_state_dict(torch.load(s['source_checkpoint'],map_location='cpu',weights_only=False)['model']);new=Tracker(False).cuda().eval().requires_grad_(False);actor=torch.load(a.actor,map_location='cpu',weights_only=False);new.load_state_dict(actor['model'])
q=load_frozen_basin(j/'fit/critic.pt','cuda');root=os.environ['DEX_YCB_DIR'];fp=FrozenFoundationPose(c['foundationpose_root'],root,torch.device('cuda',0),c['foundationpose_refiner_sha256']);renderer=Renderer('cuda');ds=ClipDataset(root,'cache/dexycb_s0',length=c['clip_length'],seed=s['seed'],steps=1_000_000,augmentation=True)
predictions=[];indices=[];cos=[]
for i,row in enumerate(meta):
 if row['split']!='test':continue
 item=ds[row['dataset_index']]
 def observe(u,inp,out):
  if u!=3:return
  oldz=torch.from_numpy(table['z'][i]).cuda()[None];reconstruction=float(torch.nn.functional.cosine_similarity(out['latent'].float(),oldz))
  if reconstruction<.999:raise RuntimeError('Saved context reconstruction mismatch')
  with torch.autocast('cuda',dtype=torch.bfloat16):z=new(**inp)['latent'].float()
  n=table['delta'].shape[1];v=q.outcomes(z.expand(n,-1),torch.from_numpy(table['delta'][i]).cuda());predictions.append({k:x.cpu().numpy() for k,x in v.items()});indices.append(i);cos.append(float(torch.nn.functional.cosine_similarity(z,oldz)))
 with torch.no_grad():batch_step(source,[item],renderer,c,4,True,False,history_mode=row['history_mode'],fp_transition=fp,observer=observe)
 if len(indices)%16==0:print(json.dumps(dict(frames=len(indices))),flush=True)
valid=table['valid'][indices];report={};trained=json.loads((j/'fit/receipt.json').read_text())
for name,target in [('converge','y_converge'),('improve','y_improve')]:
 logits=np.stack([x[name+'_logit'] for x in predictions]);p=np.clip(1/(1+np.exp(-logits)),1e-6,1-1e-6)[valid];y=table[target][indices][valid]
 report[name]=dict(auroc=float(roc_auc_score(y,p)),nll=float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),brier=float(((p-y)**2).mean()))
pairs=[]
for i,p in zip(indices,predictions):
 v=table['valid'][i];e=table['e_after'][i][v];qv=p['converge_logit'][v];mask=e[:,None]-e[None,:]>.005;diff=qv[None,:]-qv[:,None]
 if mask.any():pairs.append(float(((diff>0)+.5*(diff==0))[mask].mean()))
report.update(quality_frames=len(pairs),quality_pair_accuracy=float(np.mean(pairs)),latent_cosine_mean=float(np.mean(cos)))
checks=dict(order=report['quality_pair_accuracy']>=.6,population=report['quality_frames']>=30)
for name in ['converge','improve']:
 for field in ['nll','brier']:checks[name+'_'+field]=report[name][field]<trained['metrics']['test'][name]['constant_'+field]
 checks[name+'_auroc']=report[name]['auroc']>=.7
r=dict(passed=all(checks.values()),checks=checks,metrics=report,frames=len(indices),actor_step=actor['global_step'],actor_sha256=hashlib.sha256(Path(a.actor).read_bytes()).hexdigest(),scope='unchanged fresh labeled contexts, new actor z; no refit or threshold tuning');Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
