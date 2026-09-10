"""Write a completion receipt only after the requested optimizer step is saved."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--output',default='runs/lip_v1_s0');p.add_argument('--expected-steps',type=int,default=40000);a=p.parse_args()
out=Path(a.output);path=out/'last.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
if ck['global_step']!=a.expected_steps or ck['scheduler']['last_epoch']!=a.expected_steps:
 raise RuntimeError('Checkpoint/scheduler has not reached the requested terminal step')
if not ck['config'].get('data_complete') or ck['config'].get('diagnostic_subset'):
 raise RuntimeError('Terminal checkpoint is not a full-data formal training run')
rows=[]
for rank in range(len(ck['rng'])):
 log=out/f'rank{rank}.jsonl';last=json.loads(log.read_text().splitlines()[-1])
 if last['step']!=a.expected_steps or last['scheduler_step']!=a.expected_steps or last['nonfinite_count']:
  raise RuntimeError('Rank terminal state mismatch')
 rows.append(dict(rank=rank,step=last['step'],samples_seen=last['samples_seen']))
if len(rows)!=8:raise RuntimeError('Expected eight rank RNG states')
h=hashlib.sha256()
with path.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
r=dict(status='complete',optimizer_steps=a.expected_steps,rank_terminal_states=rows,
       checkpoint=str(path),checkpoint_sha256=h.hexdigest(),split_hash=ck['split_hash'],mesh_hash=ck['mesh_hash'],
       code_sha256=ck.get('code_sha256'),best_checkpoint_exists=(out/'best.pt').is_file(),
       scope='Training completion; does not itself assert benchmark quality')
tmp=out/'training_receipt.tmp';tmp.write_text(json.dumps(r,indent=2));tmp.replace(out/'training_receipt.json')
print(json.dumps(r,indent=2))
