"""Audit joint continuation source, Adam ownership, frozen history, and eight-rank logs."""
import argparse,json,sys
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.horizon_resume import exact
from lip.unified.reconstruction_only import is_pose_parameter
from lip.engine.jepa_checkpoint import sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args();c=yaml.safe_load(Path(a.config).read_text());r=Path(c['paths']['output'])
 source=torch.load(c['reconstruction_only']['source_checkpoint'],map_location='cpu',weights_only=False)
 initial=torch.load(r/'initial.pt',map_location='cpu',weights_only=False);last=torch.load(r/'last.pt',map_location='cpu',weights_only=False)
 assert exact(initial['model'],source['model']) and initial['step']==source['step']==35400
 def states(record):
  opt=record['optimizer'];return {name:opt['state'].get(i,{}) for group in opt['param_groups'] for name,i in zip(group['names'],group['params'])}
 old=states(source);begin=states(initial);current=states(last)
 assert all(exact(v,begin[n]) for n,v in old.items())
 new=set(begin)-set(old);assert new and all(is_pose_parameter(n) and not begin[n] for n in new)
 assert all(any(not torch.equal(v,last['model'][n]) for n,v in initial['model'].items() if n.startswith(prefix)) for prefix in ('head.','object_attn.','geometry_readout.','core.blocks.','surface_head.'))
 assert all(current[n] for n in new)
 assert all(torch.equal(v,last['model'][n]) for n,v in initial['model'].items() if n.startswith('writer.') or '.history.' in n)
 assert int(last['model']['ema_updates'])==int(initial['model']['ema_updates'])+last['step']-35400
 ranks=[]
 for rank in range(8):
  row=json.loads((r/f'rank{rank}.jsonl').read_text().splitlines()[-1]);assert row['step']>=last['step'] and row['pose_loss_enabled'] and not row['pose_parameters_frozen']
  assert row['recovery_metrics']['pose']>0;ranks.append(row['step'])
 assert last['sampler_position']==32*last['step'] and len(last['rng'])==8
 result=dict(passed=True,step=last['step'],checkpoint_sha256=sha(r/'last.pt'),initial_model_exact=True,inherited_moments_exact=True,new_pose_parameters=len(new),pose_and_jepa_updated=True,history_unchanged=True,all_rank_steps=ranks,ema_counter_continuous=True)
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
