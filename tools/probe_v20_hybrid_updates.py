"""Offline component attribution using frozen paired deltas at the same V20 base; not a deployable model."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.evaluation.metrics import errors


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);index=Path('/mnt/why/dexycb_lip/cache/dexycb_s0')
 registry={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines()) if s['split']=='val'}
 native={ (r['stream_id'],r['frame_index']):r for r in map(json.loads,Path('/mnt/why/dexycb_lip/unified_jepa_20260921/pose_geometry_v20/validation/step45400_scored/predictions.jsonl').read_text().splitlines())}
 cache={};rows=[]
 for rank in range(8):
  for row in map(json.loads,(root/f'rank{rank}/frames.jsonl').read_text().splitlines()):
   if row['base_kind']!='v20_previous':continue
   sid=row['stream_id'];f=row['frame_index'];s=registry[sid]
   if sid not in cache:
    with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
    with np.load(index/s['pose_cache']) as z:gt=z['poses'].copy()
    gt[:,:3,3]+=gt[:,:3,:3]@mesh['center'];cache[sid]=(mesh,gt)
   mesh,gt=cache[sid];base=np.array(native[sid,f-1]['pose_centered']);d=float(mesh['diameter']);v=np.array(row['deltas']['v20']);l=np.array(row['deltas']['lip_no_history'])
   variants={}
   for name,rot,trans in [('lip_rotation_v20_translation',l[:3],v[3:]),('v20_rotation_lip_translation',v[:3],l[3:])]:
    pose=base.copy();pose[:3,:3]=Rotation.from_rotvec(rot).as_matrix()@base[:3,:3];pose[:3,3]+=d*trans
    variants[name]=errors(pose,gt[f],mesh['points'],d,dtype='f4')
   rows.append(dict(physical_sequence=row['physical_sequence'],object_id=row['object_id'],visibility=row['visibility'],metrics={**{k:row['metrics'][k] for k in ['base','v20','lip_no_history']},**variants}))
 report=dict(completed=True,model_optimizer_updates=0,frames=len(rows),scope='Conditional component substitutions at the identical V20 estimated base, using frozen old-LIP no-history predictions. Diagnostic only; no hybrid inference path is installed.',populations={})
 for pop,sub in [('all',rows),('visibility_ge50',[r for r in rows if r['visibility'] is not None and r['visibility']>=.5]),('visibility_lt30',[r for r in rows if r['visibility'] is not None and r['visibility']<.3])]:
  report['populations'][pop]={n:{k:float(np.mean([r['metrics'][n][k] for r in sub])) for k in ['adds_005','rotation_deg','center_mm']} for n in rows[0]['metrics']} if sub else {}
 (root/'hybrid_updates.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['populations']['all']))
if __name__=='__main__':main()
