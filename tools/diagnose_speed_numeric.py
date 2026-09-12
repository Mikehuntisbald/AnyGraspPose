"""Measure the unchanged runtime's repeatability before judging trajectory drift."""
import json
import subprocess
import time
import yaml
from benchmark_speed import R,J,run,status

while not (J/'threaded_comparison.json').exists():time.sleep(2)
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(2)
c=yaml.safe_load((J/'baseline.yaml').read_text());c['run_id']='speed_baseline_repeat'
new=run('baseline_repeat',c,R/'runs/basin_memory_fix/candidate/src',steps=8)
old=[[json.loads(x) for x in (J/f'baseline/rank{rank}.jsonl').read_text().splitlines()][:8] for rank in range(8)]
rows=[dict(rank=rank,step=x['step'],loss_abs=abs(x['loss']-y['loss']),grad_norm_abs=abs(x['grad_norm']-y['grad_norm']))
      for rank,(a,b) in enumerate(zip(old,new)) for x,y in zip(a,b)]
r=dict(steps=8,world_size=8,same_code=True,same_sampling_and_checkpoint=True,
       max_loss_difference=max(x['loss_abs'] for x in rows),max_grad_norm_difference=max(x['grad_norm_abs'] for x in rows),
       first_gradient_difference=next((x for x in rows if x['grad_norm_abs']>0),None),
       first_loss_difference=next((x for x in rows if x['loss_abs']>0),None),rows=rows)
(J/'baseline_repeatability.json').write_text(json.dumps(r,indent=2));status('repeatability_measured',result=r)
