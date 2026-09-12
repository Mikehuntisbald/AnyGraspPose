"""Backfill continuous FP outcomes from archived poses, without rerunning FP."""
import argparse,json,hashlib,concurrent.futures
from pathlib import Path
import numpy as np
from lip.data.basin_targets import candidate_outcomes,label_policy
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--index',default='cache/dexycb_s0');p.add_argument('--workers',type=int,default=4);p.add_argument('--metric',default='adds_over_d');p.add_argument('--tau',type=float,default=.1);p.add_argument('--margin',type=float,default=.005);a=p.parse_args()
source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);index=Path(a.index)
if (out/'complete.json').exists():raise RuntimeError('Completed artifact already exists; use a new output directory')
manifest=json.loads((source/'manifest.json').read_text());original_complete=json.loads((source/'complete.json').read_text());policy=label_policy(a.metric,a.tau,a.margin)
files=sorted(source.glob('frame_[0-9][0-9][0-9][0-9][0-9].npz'));assert len(files)==original_complete['frames']
streams={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines())};meshes={}
for s in streams.values():
 if s['mesh_cache'] not in meshes:
  with np.load(index/s['mesh_cache']) as z:meshes[s['mesh_cache']]={k:z[k].copy() for k in ('vertices','center','diameter')}
new_manifest=dict(manifest,schema_version=2,label_policy=policy,source_data=str(source.resolve()),source_data_sha256=original_complete['data_sha256'],source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),backfill='before metrics recomputed from saved candidate/GT poses; after metrics copied from actual FP receipt; no FP rerun',critic_status='original convergence critic failed action-ranking gate; no prior enabled')
(out/'manifest.json').write_text(json.dumps(new_manifest,indent=2))
def enrich(file):
 with np.load(file) as z:payload={k:z[k].copy() for k in z.files}
 meta=json.loads(str(payload['meta']));mesh=meshes[streams[meta['stream_id']]['mesh_cache']]
 fields=candidate_outcomes(payload['candidate_poses'],payload['after_fp_poses'],payload['gt'],mesh['vertices'],mesh['center'],float(mesh['diameter']),after_metrics=meta['post_fp_errors'],metric=a.metric,tau=a.tau,margin=a.margin)
 if a.metric=='adds_over_d' and a.tau==.1:assert np.array_equal(fields['y_converge'],payload['labels'])
 payload.update(fields);payload['labels']=fields['y_converge'].astype('f4')
 return payload,file.read_bytes(),dict(name=file.name,candidates=len(fields['y_converge']),converge=int(fields['y_converge'].sum()),improve=int(fields['y_improve'].sum()),both=int((fields['y_converge']&fields['y_improve']).sum()),regressed=int((fields['delta_e']<0).sum()))
rows=[];payloads=[];source_hash=hashlib.sha256()
with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
 for file,result in zip(files,pool.map(enrich,files)):
  payload,raw,row=result;rows.append(row);payloads.append(payload);source_hash.update(file.name.encode());source_hash.update(raw)
  if len(rows)%128==0:print(json.dumps(dict(frames=len(rows),total=len(files))),flush=True)
assert source_hash.hexdigest()==original_complete['data_sha256']
table={k:np.stack([p[k] for p in payloads]) for k in payloads[0]};table['frame_files']=np.array([f.name for f in files])
tmp=out/'outcomes.tmp.npz';np.savez_compressed(tmp,**table);tmp.replace(out/'outcomes.npz')
digest=hashlib.sha256((out/'outcomes.npz').read_bytes()).hexdigest()
r=dict(layout='consolidated_frame_candidate_arrays',frames=len(rows),data_sha256=digest,source_data_sha256=source_hash.hexdigest(),label_policy=policy,fp_rerun=False,**{k:sum(x[k] for x in rows) for k in ('candidates','converge','improve','both','regressed')})
(out/'complete.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
