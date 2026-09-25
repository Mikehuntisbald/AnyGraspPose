"""Exploratory frozen-feature ridge readability. Physical-sequence held-out folds; no model weights updated."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);ids=[];arrays={};rows=[]
 for rank in range(8):
  r=root/f'rank{rank}';receipt=json.loads((r/'receipt.json').read_text());assert receipt['completed'] and receipt['state_digests_before']==receipt['state_digests_after']
  ids+=json.loads((r/'latent_ids.json').read_text())
  with np.load(r/'latents.npz') as z:
   for k in z.files:arrays.setdefault(k,[]).append(z[k].astype('f4'))
  rows += list(map(json.loads,(r/'frames.jsonl').read_text().splitlines()))
 arrays={k:np.concatenate(v) for k,v in arrays.items()};seqs=sorted({i['physical_sequence'] for i in ids});assert len(seqs)==40
 np.random.default_rng(42).shuffle(seqs);fold_of={sid:f for f,part in enumerate(np.array_split(seqs,5)) for sid in part};folds=np.array([fold_of[i['physical_sequence']] for i in ids])
 y=np.zeros((len(ids),6),dtype='f4');axes=[]
 for j,i in enumerate(ids):
  tag=i['tag'];axis=-1 if tag=='gt_zero' else int(tag[4]);axes.append(axis)
  if axis>=0:y[j,axis]=-int(tag.split('_')[1])
 axes=np.array(axes);nonsym=np.array([not x['symmetry'] for x in ids]);gt_ids={(i['stream_id'],i['frame']):j for j,i in enumerate(ids) if i['tag']=='gt_zero'}
 arrays['patch_minus_gt_ORACLE']=arrays['patch']-np.stack([arrays['patch'][gt_ids[i['stream_id'],i['frame']]] for i in ids])
 results={};predictions={}
 for name,x in arrays.items():
  pred=np.zeros_like(y)
  choices=[]
  for fold in range(5):
   train=folds!=fold;test=~train
   inner_sequences=sorted({ids[j]['physical_sequence'] for j in np.where(train)[0]})[:6]
   inner=np.array([i['physical_sequence'] in inner_sequences for i in ids])&train;fit=train&~inner
   def kernel_data(mask,eval_mask):
    mean=x[mask].mean(0);std=x[mask].std(0).clip(.01)
    xx=(x[mask]-mean)/std/np.sqrt(x.shape[1]);zz=(x[eval_mask]-mean)/std/np.sqrt(x.shape[1]);ym=y[mask].mean(0)
    return xx@xx.T,zz@xx.T,y[mask]-ym,ym
   kk,zk,yy,ym=kernel_data(fit,inner);scores=[]
   for penalty in (.01,.1,1.,10.):
    kreg=kk.copy();kreg.flat[::len(kreg)+1]+=penalty
    val=zk@np.linalg.solve(kreg,yy)+ym;scores.append(float(np.mean((val-y[inner])**2)))
   chosen=(.01,.1,1.,10.)[int(np.argmin(scores))];choices.append(chosen)
   kk,zk,yy,ym=kernel_data(train,test);kk.flat[::len(kk)+1]+=chosen
   pred[test]=zk@np.linalg.solve(kk,yy)+ym
  predictions[name]=pred
  rotation_pred=pred[:,:3]*np.pi/18;rotation_base=-y[:,:3]*np.pi/18
  error=np.rad2deg((Rotation.from_rotvec(rotation_pred)*Rotation.from_rotvec(rotation_base)).magnitude())
  trans_error=np.linalg.norm((pred[:,3:]-y[:,3:])*.05,axis=1)
  results[name]={'selected_penalties':choices}
  for scope,mask in [('all',np.ones(len(ids),bool)),('nonsymmetric',nonsym)]:
   rot=mask&(axes>=0)&(axes<3);tr=mask&(axes>=3);zero=mask&(axes<0)
   results[name][scope]=dict(rotation10_error_deg=float(error[rot].mean()),translation05d_error_d=float(trans_error[tr].mean()),zero_update_deg=float(error[zero].mean()),zero_translation_d=float(trans_error[zero].mean()))
 effects={}
 for arm in ('completion_off','relation_off','oracle_surface_values','oracle_surface_and_validity'):
  s=[r['effects'][arm] for r in rows if arm in r.get('effects',{})]
  effects[arm]={metric:dict(mean=float(np.mean([v[metric] for v in s])),p95=float(np.quantile([v[metric] for v in s],.95)),max=float(max(v[metric] for v in s))) for metric in ['rotation_deg','center_mm']}
 report=dict(completed=True,model_optimizer_updates=0,ridge_alpha="inner-physical-validation from0.01,0.1,1,10",folds=5,split='physical-sequence disjoint exploratory fits within existing val diagnosis',physical_sequences=40,rows=len(ids),results=results,intervention_output_changes=effects,scope='Ridge is a diagnostic fitted readout, not a production model or independent benchmark. GT-reference feature subtraction is explicitly oracle and unavailable at inference. Failure of linear decoding does not prove information is absent.')
 (root/'nested_latent_summary.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(root/'nested_ridge_predictions.npz',**predictions);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
