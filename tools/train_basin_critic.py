"""Fit, calibrate, and gate a frozen-scene candidate critic on disjoint train-sequence groups."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch,yaml
from sklearn.metrics import roc_auc_score
from lip.models.basin import BasinCritic
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
data=Path(a.data);out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
assert (data/'complete.json').exists();manifest=json.loads((data/'manifest.json').read_text());spec=manifest['spec']
zs=[];ds=[];ys=[];metas=[]
for f in sorted(data.glob('frame_[0-9][0-9][0-9][0-9][0-9].npz')):
 with np.load(f) as z:zs.append(z['z']);ds.append(z['delta']);ys.append(z['labels']);metas.append(json.loads(str(z['meta'])))
z=torch.tensor(np.stack(zs));delta=torch.tensor(np.stack(ds));y=torch.tensor(np.stack(ys));splits={s:np.array([i for i,m in enumerate(metas) if m['split']==s]) for s in ['train','calibration','test']}
groups={s:{metas[i]['physical_sequence'] for i in ids} for s,ids in splits.items()};assert not (groups['train']&groups['calibration'] or groups['train']&groups['test'] or groups['calibration']&groups['test'])
assert len(z)==spec['frames'] and y.shape[1]==spec['candidates_per_frame']
torch.set_num_threads(2);torch.manual_seed(spec['seed']);device='cuda' if torch.cuda.is_available() else 'cpu';model=BasinCritic().to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay']);g=torch.Generator().manual_seed(spec['seed']);best=float('inf');stall=0

def logits(ids):
 with torch.no_grad():
  zz=z[ids].to(device)[:,None].expand(-1,6,-1);dd=delta[ids].to(device);return model(zz.reshape(-1,256),dd.reshape(-1,6)).reshape(-1,6)

with (out/'training.jsonl').open('w') as log:
 for epoch in range(spec['max_epochs']):
  model.train();order=splits['train'][torch.randperm(len(splits['train']),generator=g).numpy()];trainloss=[]
  for i in range(0,len(order),spec['batch_frames']):
   ids=order[i:i+spec['batch_frames']];zz=z[ids].to(device)[:,None].expand(-1,6,-1);dd=delta[ids].to(device);yy=y[ids].to(device)
   pred=model(zz.reshape(-1,256),dd.reshape(-1,6)).reshape_as(yy);loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,yy)
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();trainloss.append(float(loss))
  model.eval();v=float(torch.nn.functional.binary_cross_entropy_with_logits(logits(splits['calibration']),y[splits['calibration']].to(device)))
  r=dict(epoch=epoch,train_nll=float(np.mean(trainloss)),calibration_nll=v);log.write(json.dumps(r)+'\n');log.flush()
  if v<best-1e-5:best=v;stall=0;best_state={k:t.cpu().clone() for k,t in model.state_dict().items()};best_epoch=epoch
  else:stall+=1
  if epoch%10==0:print(json.dumps(r),flush=True)
  if stall>=spec['patience']:break
model.load_state_dict(best_state);model.eval();raw=logits(splits['calibration']).detach();target=y[splits['calibration']].to(device)
# Temperature selection uses calibration groups only, never test groups.
temps=torch.logspace(-.6,.6,81,device=device);nll=[float(torch.nn.functional.binary_cross_entropy_with_logits(raw/t,target)) for t in temps];temperature=float(temps[int(np.argmin(nll))]);model.temperature.fill_(temperature)
prior=float(y[splits['train']].mean())
def metrics(ids,pred):
 yy=y[ids].numpy();pp=np.clip(pred,1e-6,1-1e-6);pos=int(yy.sum());neg=int(yy.size-pos);mixed=[]
 for yi,pi in zip(yy,pp):
  if yi.min()!=yi.max():
   d=pi[yi==1,None]-pi[None,yi==0];mixed.append(float(((d>0)+.5*(d==0)).mean()))
 pflat=pp.ravel();yflat=yy.ravel();ece=0.
 for lo in np.linspace(0,.9,10):
  mask=(pflat>=lo)&(pflat<lo+.1)
  if mask.any():ece+=float(mask.mean()*abs(pflat[mask].mean()-yflat[mask].mean()))
 return dict(frames=len(ids),positive=pos,negative=neg,positive_rate=float(yy.mean()),auroc=float(roc_auc_score(yy.ravel(),pp.ravel())) if pos and neg else None,
             nll=float(-(yy*np.log(pp)+(1-yy)*np.log(1-pp)).mean()),brier=float(((pp-yy)**2).mean()),ece10=ece,
             mixed_frames=len(mixed),pair_accuracy=float(np.mean(mixed)) if mixed else None,
             constant_nll=float(-(yy*np.log(prior)+(1-yy)*np.log(1-prior)).mean()),constant_brier=float(((prior-yy)**2).mean()))
reports={s:metrics(ids,torch.sigmoid(logits(ids)).cpu().numpy()) for s,ids in splits.items()}
t=reports['test'];checks=dict(enough_positive=t['positive']>=spec['min_test_positive'],enough_negative=t['negative']>=spec['min_test_negative'],enough_mixed=t['mixed_frames']>=spec['min_test_mixed_frames'],auroc=t['auroc'] is not None and t['auroc']>=spec['min_test_auroc'],pair=t['pair_accuracy'] is not None and t['pair_accuracy']>=spec['min_test_pair_accuracy'],brier=t['brier']<t['constant_brier'],nll=t['nll']<t['constant_nll'])
receipt=dict(data_sha256=json.loads((data/'complete.json').read_text())['data_sha256'],passed=all(checks.values()),checks=checks,metrics=reports,best_epoch=best_epoch,temperature=temperature,scope='physical-sequence-held-out groups within official train; no official val/test accessed',spec=spec,source_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),source_checkpoint_sha256=manifest['source_checkpoint_sha256'])
model.requires_grad_(False);torch.save(dict(model=model.cpu().state_dict(),receipt=receipt),out/'critic.pt');receipt['critic_sha256']=hashlib.sha256((out/'critic.pt').read_bytes()).hexdigest();(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)
