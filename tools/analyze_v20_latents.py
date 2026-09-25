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
  for fold in range(5):
   train=folds!=fold;test=~train;mean=x[train].mean(0);std=x[train].std(0).clip(.01)
   xx=(x[train]-mean)/std/np.sqrt(x.shape[1]);zz=(x[test]-mean)/std/np.sqrt(x.shape[1]);ym=y[train].mean(0)
   kernel=xx@xx.T;kernel.flat[::len(kernel)+1]+=.01
   alpha=np.linalg.solve(kernel,y[train]-ym);pred[test]=zz@xx.T@alpha+ym
  predictions[name]=pred
  rotation_pred=pred[:,:3]*np.pi/18;rotation_base=-y[:,:3]*np.pi/18
  error=np.rad2deg((Rotation.from_rotvec(rotation_pred)*Rotation.from_rotvec(rotation_base)).magnitude())
  trans_error=np.linalg.norm((pred[:,3:]-y[:,3:])*.05,axis=1)
  results[name]={}
  for scope,mask in [('all',np.ones(len(ids),bool)),('nonsymmetric',nonsym)]:
   rot=mask&(axes>=0)&(axes<3);tr=mask&(axes>=3);zero=mask&(axes<0)
   results[name][scope]=dict(rotation10_error_deg=float(error[rot].mean()),translation05d_error_d=float(trans_error[tr].mean()),zero_update_deg=float(error[zero].mean()),zero_translation_d=float(trans_error[zero].mean()))
 effects={}
 for arm in ('completion_off','relation_off','oracle_surface_values','oracle_surface_and_validity'):
  s=[r['effects'][arm] for r in rows if arm in r.get('effects',{})]
  effects[arm]={metric:dict(mean=float(np.mean([v[metric] for v in s])),p95=float(np.quantile([v[metric] for v in s],.95)),max=float(max(v[metric] for v in s))) for metric in ['rotation_deg','center_mm']}
 report=dict(completed=True,model_optimizer_updates=0,ridge_alpha=.01,folds=5,split='physical-sequence disjoint exploratory fits within existing val diagnosis',physical_sequences=40,rows=len(ids),results=results,intervention_output_changes=effects,scope='Ridge is a diagnostic fitted readout, not a production model or independent benchmark. GT-reference feature subtraction is explicitly oracle and unavailable at inference. Failure of linear decoding does not prove information is absent.')
 (root/'latent_summary.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(root/'ridge_predictions.npz',**predictions);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
