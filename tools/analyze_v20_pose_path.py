"""Summarize frozen pose diagnostics, explicitly separating native and GT-oracle arms."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);rows=[];receipts=[]
 for rank in range(8):
  r=root/f'rank{rank}';d=json.loads((r/'receipt.json').read_text());assert d['completed'] and d['state_digests_before']==d['state_digests_after'];receipts.append(d);rows += [json.loads(x) for x in (r/'frames.jsonl').read_text().splitlines()]
 def aggregate(selected):
  names=set.intersection(*(set(r['metrics']) for r in selected));out={}
  for name in sorted(names):
   out[name]={k:float(np.mean([r['metrics'][name][k] for r in selected])) for k in ('rotation_deg','center_mm','adds_005')}
  return dict(rows=len(selected),physical_sequences=len({r['physical_sequence'] for r in selected}),models=out)
 groups={}
 for pop,filt in [('all',lambda r:True),('nonsymmetric',lambda r:not r['symmetry']),('visible_ge50',lambda r:r['visibility'] is not None and r['visibility']>=.5),('visibility_lt30',lambda r:r['visibility'] is not None and r['visibility']<.3)]:
  for kind,match in [('natural_lip_base',lambda x:x=='lip_previous'),('natural_v20_base',lambda x:x=='v20_previous'),('correct_pose',lambda x:x=='gt_zero'),('rotation10',lambda x:x.startswith(('axis0','axis1','axis2'))),('translation05d',lambda x:x.startswith(('axis3','axis4','axis5')))]:
   sub=[r for r in rows if filt(r) and match(r['base_kind'])]
   if sub:groups[pop+'/'+kind]=aggregate(sub)
 pairs={}
 for row in rows:
  if row['base_kind'].startswith('axis'):
   key=(row['stream_id'],row['frame_index'],int(row['base_kind'][4]));pairs.setdefault(key,{})[row['base_kind'].split('_')[1]]=row
 gains=[]
 for (sid,f,axis),pair in pairs.items():
  pos,neg=pair['+1'],pair['-1'];scale=np.pi/18 if axis<3 else .05
  for model in ('v20','lip_no_history','lip_history'):
   gain=-(pos['deltas'][model][axis]-neg['deltas'][model][axis])/(2*scale)
   gains.append(dict(stream_id=sid,physical_sequence=pos['physical_sequence'],frame_index=f,axis=axis,model=model,gain=gain,symmetry=pos['symmetry']))
 gain_summary={}
 for sym in (False,True):
  for axis in range(6):
   gain_summary[f'{"all" if sym else "nonsymmetric"}/axis{axis}']={m:float(np.mean([r['gain'] for r in gains if r['model']==m and r['axis']==axis and (sym or not r['symmetry'])])) for m in ('v20','lip_no_history','lip_history')}
 oracle={}
 for kind in ('lip_previous','v20_previous','gt_zero','axis0_+1','axis1_+1','axis2_+1'):
  sub=[r for r in rows if r['base_kind']==kind];oracle[kind]=aggregate(sub)
 summary=dict(completed=True,optimizer_updates=0,physical_sequences=len({r['physical_sequence'] for r in rows}),sampled_frames=len({(r['stream_id'],r['frame_index']) for r in rows}),objects=len({r['object_id'] for r in rows}),groups=groups,gains=gain_summary,readout_interventions=oracle,
  max_crop_difference=max(r['maximum_crop_difference'] for r in receipts),max_old_lip_replay_difference=max(r['old_lip_replay_max_abs'] for r in receipts),max_v20_native_replay_difference=max(r['native_pose_max_abs'] for r in rows if r['native_pose_max_abs'] is not None),
  average_completion_weight=float(np.mean([r['completed_weight'] for r in rows])),relation_gain=rows[0]['relation_gain'],scope=receipts[0]['scope'])
 (root/'pose_path_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'gain_rows.json').write_text(json.dumps(gains))
 print(json.dumps({k:groups[k] for k in ['nonsymmetric/rotation10','all/correct_pose','all/natural_v20_base','all/natural_lip_base']},indent=2));print('GAINS',json.dumps(gain_summary))
if __name__=='__main__':main()
