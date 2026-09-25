"""Read-only paired native-rollout failure decomposition; no training or test split."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.jepa_checkpoint import sha

ROOT=Path('/mnt/why/dexycb_lip')
PATHS={'lip':ROOT/'smooth_val_evaluation_20260915/runs/full/residual/scored',
 'v20':ROOT/'unified_jepa_20260921/pose_geometry_v20/validation/step45400_scored'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 index=ROOT/'cache/dexycb_s0';registry={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines()) if s['split']=='val'}
 manifests={k:json.loads((v/'manifest.json').read_text()) for k,v in PATHS.items()}
 for key in ('split_hash','mesh_hash','initializers_sha256','frames'):assert manifests['lip'][key]==manifests['v20'][key]
 models={k:{(r['stream_id'],r['frame_index']):r for r in map(json.loads,(v/'predictions.jsonl').read_text().splitlines())} for k,v in PATHS.items()}
 assert models['lip'].keys()==models['v20'].keys()
 records=[]
 def rotation(a,b):return float(np.rad2deg(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))
 for sid,s in registry.items():
  with np.load(index/s['mesh_cache']) as z:center=z['center'].copy();d=float(z['diameter'])
  with np.load(index/s['pose_cache']) as z:truth=z['poses'].copy()
  truth[:,:3,3]+=truth[:,:3,:3]@center
  for f in range(s['num_frames']):
   key=(sid,f);base=models['lip'][key]
   if base['pose_centered'] is None or base['initialization']:continue
   row=dict(stream_id=sid,physical_sequence='/'.join(sid.split('/')[:2]),frame_index=f,object_id=s['object_id'],visibility=base['visibility'],updates=base['updates_after_initialization'],moving=base['moving'],diameter=d)
   gt=truth[f]
   for name in models:
    r=models[name][key];prev=models[name].get((sid,f-1));pose=np.asarray(r['pose_centered']);rot=rotation(pose[:3,:3],gt[:3,:3]);center_err=float(np.linalg.norm(pose[:3,3]-gt[:3,3])/d)
    v=dict(adds005=r['adds_005'],center_d=center_err,rotation_deg=rot,translation_ok=center_err<.05)
    if prev and prev['pose_centered'] is not None:
     before=np.asarray(prev['pose_centered']);be=float(np.linalg.norm(before[:3,3]-gt[:3,3])/d);br=rotation(before[:3,:3],gt[:3,:3]);
     v.update(base_center_d=be,base_rotation_deg=br,center_change_d=center_err-be,rotation_change_deg=rot-br,
      rotation_update_deg=rotation(pose[:3,:3],before[:3,:3]),center_update_d=float(np.linalg.norm(pose[:3,3]-before[:3,3])/d),previous_success=prev['adds_005'])
    row[name]=v
   records.append(row)
 def stats(rows,name):
  groups={}
  for r in rows:groups.setdefault(r['object_id'],[]).append(r[name])
  keys=['adds005','center_d','rotation_deg','translation_ok','center_change_d','rotation_change_deg','rotation_update_deg','center_update_d']
  return dict(frames=len(rows),objects=len(groups),**{k:float(np.mean([np.mean([r[k] for r in g if k in r]) for g in groups.values()])) for k in keys})
 selections={'all_initialized':records,'visible_ge50':[r for r in records if r['visibility'] is not None and r['visibility']>=.5],'visibility_lt50':[r for r in records if r['visibility'] is not None and r['visibility']<.5],'visibility_lt30':[r for r in records if r['visibility'] is not None and r['visibility']<.3],
  'first8':[r for r in records if r['updates']<=8],'after32':[r for r in records if r['updates']>32], 'moving':[r for r in records if r['moving'] is True],'static':[r for r in records if r['moving'] is False]}
 report={pop:{name:stats(rows,name) for name in models} for pop,rows in selections.items() if rows}
 losses={}
 for name in models:
  eligible=[r for r in records if 'previous_success' in r[name] and r[name]['previous_success']]
  lost=[r for r in eligible if not r[name]['adds005']]
  losses[name]=dict(previous_success_frames=len(eligible),lost_next_frame=len(lost),loss_rate=len(lost)/len(eligible))
 object_rows=[]
 for oid in sorted({r['object_id'] for r in records}):
  sub=[r for r in records if r['object_id']==oid];values={n:stats(sub,n) for n in models};object_rows.append(dict(object_id=oid,**values))
 receipt=dict(completed=True,training=False,optimizer_updates=0,official_test_access=False,source_checkpoints={k:v['checkpoint_sha256'] for k,v in manifests.items()},protocol_match=True,scope='Canonical rotation is not symmetry reduced; ADD-S is primary. Initialized frames only in decomposition; official all-frame scores retained separately.',populations=report,success_loss=losses,objects=object_rows)
 (out/'native_decomposition.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'native_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
