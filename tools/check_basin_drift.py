"""Check a frozen critic under a newer actor's latent on unchanged labeled contexts."""
import argparse,json,os,hashlib
from pathlib import Path
import numpy as np,torch,yaml
from sklearn.metrics import roc_auc_score
from lip.models.tracker import Tracker
from lip.models.basin import BasinCritic
from lip.data.clips import ClipDataset
from lip.engine.runtime import batch_step
from lip.geometry.renderer import Renderer
from lip.integrations.frozen_fp import FrozenFoundationPose
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--critic',required=True);p.add_argument('--actor',required=True);p.add_argument('--out',required=True);a=p.parse_args()
data=Path(a.data);m=json.loads((data/'manifest.json').read_text());spec=m['spec'];c=yaml.safe_load(Path(spec['source_config']).read_text());torch.set_num_threads(2);torch.manual_seed(spec['seed'])
source=Tracker(False).cuda().eval().requires_grad_(False);source.load_state_dict(torch.load(spec['source_checkpoint'],map_location='cpu',weights_only=False)['model'])
new=Tracker(False).cuda().eval().requires_grad_(False);newck=torch.load(a.actor,map_location='cpu',weights_only=False);new.load_state_dict(newck['model'])
q=BasinCritic().cuda().eval().requires_grad_(False);qc=torch.load(a.critic,map_location='cpu',weights_only=False);assert qc['receipt']['passed'];q.load_state_dict(qc['model'])
root=os.environ['DEX_YCB_DIR'];ds=ClipDataset(root,'cache/dexycb_s0',length=c['clip_length'],seed=spec['seed'],steps=spec['frames']*10,augmentation=True);fp=FrozenFoundationPose(c['foundationpose_root'],root,torch.device('cuda',0),c['foundationpose_refiner_sha256']);renderer=Renderer('cuda')
labels=[];probs=[];cosines=[]
for file in sorted(data.glob('frame_[0-9][0-9][0-9][0-9][0-9].npz')):
 with np.load(file) as z:
  meta=json.loads(str(z['meta']));d=z['delta'].copy();target=z['labels'].copy();oldz=z['z'].copy()
 if meta['split']!='test':continue
 item=ds[meta['dataset_index']]
 def observe(u,inp,pred):
  if u!=3:return
  with torch.autocast('cuda',dtype=torch.bfloat16):now=new(**inp)['latent'].float()
  logits=q(now.expand(6,-1),torch.from_numpy(d).cuda());probs.append(torch.sigmoid(logits).cpu().numpy());labels.append(target);cosines.append(float(torch.nn.functional.cosine_similarity(now,torch.from_numpy(oldz).cuda()[None])))
 with torch.no_grad():batch_step(source,[item],renderer,c,4,True,False,history_mode=meta['history_mode'],fp_transition=fp,observer=observe)
 if len(labels)%16==0:print(json.dumps(dict(frames=len(labels))),flush=True)
y=np.stack(labels);pr=np.clip(np.stack(probs),1e-6,1-1e-6);pairs=[]
for yi,pi in zip(y,pr):
 if yi.min()!=yi.max():
  d=pi[yi==1,None]-pi[None,yi==0];pairs.append(float(((d>0)+.5*(d==0)).mean()))
t=qc['receipt']['metrics']['test'];metrics=dict(auroc=float(roc_auc_score(y.ravel(),pr.ravel())),nll=float(-(y*np.log(pr)+(1-y)*np.log(1-pr)).mean()),brier=float(((pr-y)**2).mean()),pair_accuracy=float(np.mean(pairs)),mean_latent_cosine=float(np.mean(cosines)))
passed=metrics['auroc']>=spec['min_test_auroc'] and metrics['pair_accuracy']>=spec['min_test_pair_accuracy'] and metrics['nll']<t['constant_nll'] and metrics['brier']<t['constant_brier']
r=dict(passed=passed,actor_step=newck['global_step'],actor_sha256=hashlib.sha256(Path(a.actor).read_bytes()).hexdigest(),frames=len(labels),metrics=metrics,scope='same fixed held-out labeled contexts; no refit; tests conditioning drift, not actor benefit')
Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
