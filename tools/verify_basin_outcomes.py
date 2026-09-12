"""Audit rich outcome schema, lineage and a seeded independent metric recomputation."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from lip.data.basin_targets import candidate_outcomes
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--index',default='cache/dexycb_s0');p.add_argument('--samples',type=int,default=32);a=p.parse_args()
data=Path(a.data);manifest=json.loads((data/'manifest.json').read_text());source=Path(manifest['source_data']);index=Path(a.index)
with np.load(data/'outcomes.npz') as raw:table={k:raw[k].copy() for k in raw.files}
files=[source/str(n) for n in table['frame_files']];assert len(files)==json.loads((data/'complete.json').read_text())['frames']
selected=set(np.random.default_rng(731).choice(len(files),min(a.samples,len(files)),replace=False).tolist());streams={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines())}
required=['candidate_pose','refined_pose','add_before_over_d','add_after_over_d','adds_before_over_d','adds_after_over_d','rotation_before_deg','rotation_after_deg','center_before_mm','center_after_mm','e_before','e_after','delta_e','y_converge','y_improve']
h=hashlib.sha256((data/'outcomes.npz').read_bytes());source_hash=hashlib.sha256();cases=0;cross=np.zeros((2,2),dtype=int);regressed=0
for i,f in enumerate(files):
 orig=source/f.name;source_hash.update(f.name.encode());source_hash.update(orig.read_bytes())
 z={k:v[i] for k,v in table.items()}
 with np.load(orig) as old:
  assert all(k in z for k in required)
  assert np.array_equal(z['candidate_pose'],old['candidate_poses']) and np.array_equal(z['refined_pose'],old['after_fp_poses'])
  assert np.array_equal(z['y_converge'],old['labels'])
  assert np.array_equal(z['delta_e'],z['e_before']-z['e_after'])
  assert np.array_equal(z['y_converge'],z['e_after']<float(z['target_tau']))
  assert np.array_equal(z['y_improve'],z['e_after']<z['e_before']-float(z['target_margin']))
  cases+=len(z['y_converge']);regressed+=int((z['delta_e']<0).sum())
  for c,m in zip(z['y_converge'],z['y_improve']):cross[int(c),int(m)]+=1
  if i in selected:
   meta=json.loads(str(z['meta']));s=streams[meta['stream_id']]
   with np.load(index/s['mesh_cache']) as mesh:
    check=candidate_outcomes(z['candidate_pose'],z['refined_pose'],z['gt'],mesh['vertices'],mesh['center'],float(mesh['diameter']),metric=str(z['target_metric']),tau=float(z['target_tau']),margin=float(z['target_margin']))
   for k in required:
    assert np.allclose(z[k],check[k],rtol=1e-8,atol=1e-10),k
receipt=json.loads((data/'complete.json').read_text());assert h.hexdigest()==receipt['data_sha256'];assert source_hash.hexdigest()==receipt['source_data_sha256']
r=dict(passed=True,frames=len(files),candidates=cases,recomputed_frames=len(selected),recomputed_candidates=len(selected)*6,source_unchanged=True,convergence_labels_unchanged=True,regressed=regressed,joint_labels={'converge_0_improve_0':int(cross[0,0]),'converge_0_improve_1':int(cross[0,1]),'converge_1_improve_0':int(cross[1,0]),'converge_1_improve_1':int(cross[1,1])},data_sha256=h.hexdigest())
(data/'verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
