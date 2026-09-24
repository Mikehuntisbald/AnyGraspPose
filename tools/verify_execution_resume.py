"""Verify optimizer/RNG-preserving execution migration and frozen branches."""
import argparse,json,sys
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.horizon_resume import exact
from lip.unified.reconstruction_only import is_pose_parameter
from lip.engine.jepa_checkpoint import sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 c=yaml.safe_load(Path(a.config).read_text());r=Path(c['paths']['output']);plan=c['performance_resume']
 source=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)
 first=torch.load(r/'initial.pt',map_location='cpu',weights_only=False);last=torch.load(r/'last.pt',map_location='cpu',weights_only=False)
 for key in ('model','optimizer','scheduler'):assert exact(source[key],first[key]),key
 assert first['step']==source['step'] and first['sampler_position']==source['sampler_position']
 assert len(first['rng'])==len(last['rng'])==8 and last['sampler_position']==last['step']*32
 import numpy as np
 for initial,old in zip(first['rng'],source['rng']):
  assert initial['python']==old['python'] and np.array_equal(initial['numpy'][1],old['numpy'][1]) and initial['numpy'][2:]==old['numpy'][2:]
  assert exact(initial['torch'],old['torch']) and exact(initial['cuda'],old['cuda'])
 for name,v in source['model'].items():
  if is_pose_parameter(name) or name.startswith('writer.') or '.history.' in name:assert torch.equal(v,last['model'][name]),name
 assert int(last['model']['ema_updates'])==int(source['model']['ema_updates'])+last['step']-source['step']
 from lip.unified.horizon_resume import extension_factor,scheduler_origin
 import math
 assert last['scheduler']['last_epoch']==last['step']-scheduler_origin(c)
 h=c['horizon_continuation']
 for g,base in zip(last['optimizer']['param_groups'],last['scheduler']['base_lrs']):
  assert math.isclose(g['lr'],base*extension_factor(last['step'],h['source_step'],h['rewarm_steps'],c['training']['max_steps'],h['floor']),rel_tol=1e-10)
 steps=[]
 for rank in range(8):
  row=json.loads((r/f'rank{rank}.jsonl').read_text().splitlines()[-1]);steps.append(row['step'])
  assert row['pose_parameters_frozen'] and not row['pose_loss_enabled']
  assert all(math.isfinite(v) for v in row['recovery_metrics'].values())
 assert steps==[last['step']]*8
 result=dict(passed=True,step=last['step'],source_step=source['step'],all_rank_steps=steps,
  model_optimizer_scheduler_rng_exact_at_migration=True,optimizer_reset=False,ema_updates=int(last['model']['ema_updates']),
  frozen_pose_history_exact=True,schedule_unchanged=True,checkpoint_sha256=sha(r/'last.pt'))
 Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
