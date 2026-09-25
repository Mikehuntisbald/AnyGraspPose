"""Recreate representation-response and train-base summaries from frozen probe records."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);ids=[];arrays={};rows=[]
 for rank in range(8):
  r=root/'latent'/f'rank{rank}';ids+=json.loads((r/'latent_ids.json').read_text())
  with np.load(r/'latents.npz') as z:
   for k in z.files:arrays.setdefault(k,[]).append(z[k].astype('f4'))
  rows+=json.loads((root/'training_bases'/f'rank{rank}/rows.json').read_text())
 arrays={k:np.concatenate(v) for k,v in arrays.items()};lookup={(x['stream_id'],x['frame'],x['tag']):i for i,x in enumerate(ids)};response={}
 for name,x in arrays.items():
  vals={'rotation':[],'translation':[]}
  for i,a in enumerate(ids):
   if a['symmetry'] or not a['tag'].endswith('_+1'):continue
   j=lookup[a['stream_id'],a['frame'],a['tag'].replace('_+1','_-1')];axis=int(a['tag'][4]);p,q=x[i],x[j]
   relative=np.linalg.norm(p-q)/max(.5*(np.linalg.norm(p)+np.linalg.norm(q)),1e-8);cos=np.dot(p,q)/max(np.linalg.norm(p)*np.linalg.norm(q),1e-8)
   vals['rotation' if axis<3 else 'translation'].append((float(relative),float(cos)))
  response[name]={kind:dict(relative_l2=float(np.mean([v[0] for v in pairs])),cosine=float(np.mean([v[1] for v in pairs]))) for kind,pairs in vals.items()}
 (root/'latent/representation_response.json').write_text(json.dumps(response,indent=2))
 assert len(rows)==2496;summary={}
 for stage,match in [('all',lambda x:True),('first7',lambda x:x['frame']<8),('later',lambda x:x['frame']>=8)]:
  summary[stage]={}
  for arm in ['fixed_reference','student_feedback']:
   selected=[x for x in rows if x['arm']==arm and match(x)];v=np.array([x['base_rotation_deg'] for x in selected]);t=np.array([x['base_center_d'] for x in selected]);summary[stage][arm]=dict(frames=len(selected),rotation_mean=float(v.mean()),rotation_median=float(np.median(v)),rotation_lt5=float((v<5).mean()),rotation_gt30=float((v>30).mean()),center_mean_d=float(t.mean()),center_median_d=float(np.median(t)))
 (root/'training_bases/summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
