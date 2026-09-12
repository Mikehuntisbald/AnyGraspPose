import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch,yaml
from sklearn.metrics import roc_auc_score
from lip.models.basin import BasinOutcomeCritic
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=yaml.safe_load(Path(a.spec).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
data=Path(s['data']);receipt=json.loads((data/'verification.json').read_text());assert receipt['passed'];manifest=json.loads((data/'manifest.json').read_text())
with np.load(data/'outcomes.npz') as f:
 z=torch.from_numpy(f['z'].copy());delta=torch.from_numpy(f['delta'].copy());conv=torch.from_numpy(f['y_converge'].astype('f4'));imp=torch.from_numpy(f['y_improve'].astype('f4'));after=torch.from_numpy(f['e_after'].astype('f4'));meta=[json.loads(str(x)) for x in f['meta']];valid=torch.from_numpy(f['valid'].copy()) if 'valid' in f else torch.ones_like(conv,dtype=torch.bool)
idx={k:np.array([i for i,m in enumerate(meta) if m['split']==k]) for k in ['train','calibration','test']}
groups={k:{meta[i]['physical_sequence'] for i in ids} for k,ids in idx.items()};assert not(groups['train']&groups['test'] or groups['calibration']&groups['test'] or groups['train']&groups['calibration'])
NC=delta.shape[1]
torch.set_num_threads(2);torch.manual_seed(s['seed']);device='cuda';q=BasinOutcomeCritic().to(device)
actor=torch.load(s['source_actor'],map_location='cpu',weights_only=False);q.reference.load_state_dict({k.removeprefix('head.'):v for k,v in actor['model'].items() if k.startswith('head.')});del actor
opt=torch.optim.AdamW([p for p in q.parameters() if p.requires_grad],lr=s['learning_rate'],weight_decay=s['weight_decay']);gen=torch.Generator().manual_seed(s['seed']);best=float('inf');stall=0

def predict(ids):
 n=len(ids);return {k:v.reshape(n,NC) for k,v in q.outcomes(z[ids].to(device)[:,None].expand(-1,NC,-1).reshape(-1,256),delta[ids].to(device).reshape(-1,6)).items()}
def objective(ids,p):
 yc=conv[ids].to(device);yi=imp[ids].to(device);e=after[ids].to(device);qc=p['converge_logit'];gap=e[:,:,None]-e[:,None,:];ok=valid[ids].to(device);mask=(gap>s['pair_margin_d']) & ok[:,:,None] & ok[:,None,:]
 # i has larger post-FP error than j, so its quality/convergence score should be lower.
 dif=qc[:,:,None]-qc[:,None,:];pair=torch.nn.functional.softplus(dif)[mask].mean() if mask.any() else qc.sum()*0
 return torch.nn.functional.binary_cross_entropy_with_logits(qc[ok],yc[ok])+torch.nn.functional.binary_cross_entropy_with_logits(p['improve_logit'][ok],yi[ok])+s['log_after_weight']*torch.nn.functional.smooth_l1_loss(p['log_after'][ok],torch.log(e[ok].clamp_min(1e-5)))+s['pair_weight']*pair
with (out/'train.jsonl').open('w') as log:
 for epoch in range(s['max_epochs']):
  q.train();order=idx['train'][torch.randperm(len(idx['train']),generator=gen).numpy()];ls=[]
  for start in range(0,len(order),s['batch_frames']):
   ids=order[start:start+s['batch_frames']];loss=objective(ids,predict(ids));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),5.,error_if_nonfinite=True);opt.step();ls.append(float(loss.detach()))
  q.eval()
  with torch.no_grad():v=float(objective(idx['calibration'],predict(idx['calibration'])))
  r=dict(epoch=epoch,train_loss=float(np.mean(ls)),calibration_objective=v);log.write(json.dumps(r)+'\n');log.flush()
  if epoch%10==0:print(json.dumps(r),flush=True)
  if v<best-1e-5:best=v;stall=0;state={k:v.cpu().clone() for k,v in q.state_dict().items()};best_epoch=epoch
  else:stall+=1
  if stall>=s['patience']:break
q.load_state_dict(state);q.eval()
with torch.no_grad():
 raw=predict(idx['calibration']);temps=torch.logspace(-.6,.6,81,device=device)
 for h,(key,y) in enumerate([('converge_logit',conv),('improve_logit',imp)]):
  losses=[float(torch.nn.functional.binary_cross_entropy_with_logits((raw[key]/t)[valid[idx['calibration']].to(device)],y[idx['calibration']].to(device)[valid[idx['calibration']].to(device)])) for t in temps];q.temperatures[h]=temps[int(np.argmin(losses))]

def metrics(ids):
 with torch.no_grad():p={k:v.cpu().numpy() for k,v in predict(ids).items()}
 result={}
 for key,ys in [('converge',conv),('improve',imp)]:
  ok=valid[ids].numpy();y=ys[ids].numpy()[ok];prob=np.clip(1/(1+np.exp(-p[key+'_logit'])),1e-6,1-1e-6)[ok];prior=float(ys[idx['train']][valid[idx['train']]].mean());result[key]=dict(auroc=float(roc_auc_score(y.ravel(),prob.ravel())),nll=float(-(y*np.log(prob)+(1-y)*np.log(1-prob)).mean()),brier=float(((prob-y)**2).mean()),constant_nll=float(-(y*np.log(prior)+(1-y)*np.log(1-prior)).mean()),constant_brier=float(((prior-y)**2).mean()),positive=int(y.sum()),negative=int(y.size-y.sum()))
 pairs=[];binary=[];n_pairs=0
 for e,score,y,ok in zip(after[ids].numpy(),p['converge_logit'],conv[ids].numpy(),valid[ids].numpy()):
  e=e[ok];score=score[ok];y=y[ok]
  mask=e[:,None]-e[None,:]>s['pair_margin_d'];d=score[None,:]-score[:,None]
  if mask.any():pairs.append(float(((d>0)+.5*(d==0))[mask].mean()));n_pairs+=int(mask.sum())
  if y.min()!=y.max():
   d=score[y==1,None]-score[None,y==0];binary.append(float(((d>0)+.5*(d==0)).mean()))
 result.update(frames=len(ids),quality_frames=len(pairs),quality_pairs=n_pairs,quality_pair_accuracy=float(np.mean(pairs)),binary_mixed_frames=len(binary),binary_pair_accuracy=float(np.mean(binary)) if binary else None,log_after_mae=float(np.abs(p['log_after']-np.log(after[ids].numpy().clip(1e-5)))[valid[ids].numpy()].mean()))
 return result
reports={k:metrics(ids) for k,ids in idx.items()};t=reports['test'];checks={}
for key in ['converge','improve']:
 checks[key+'_auroc']=t[key]['auroc']>=s['min_auroc'];checks[key+'_nll']=t[key]['nll']<t[key]['constant_nll'];checks[key+'_brier']=t[key]['brier']<t[key]['constant_brier']
checks.update(quality_population=t['quality_frames']>=s['min_quality_frames'],quality_order=t['quality_pair_accuracy']>=s['min_quality_pair_accuracy'])
r=dict(passed=all(checks.values()),checks=checks,metrics=reports,spec=s,best_epoch=best_epoch,temperatures=q.temperatures.cpu().tolist(),data_sha256=receipt['data_sha256'],source_actor_sha256=hashlib.sha256(Path(s['source_actor']).read_bytes()).hexdigest(),acceptance_change=s.get('validation_scope','continuous post-FP quality ordering; previous failed critics retained'))
q.eval().requires_grad_(False);torch.save(dict(architecture='outcome_v2',model=q.cpu().state_dict(),receipt=r),out/'critic.pt');r['critic_sha256']=hashlib.sha256((out/'critic.pt').read_bytes()).hexdigest();(out/'receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
