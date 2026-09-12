"""Isolate the exact threaded-decoder change from the GPU upload experiment."""
import hashlib
import json
import torch
import yaml
from benchmark_speed import R,J,run,timing,status

c=yaml.safe_load((J/'optimized.yaml').read_text());c['preload_rollout_observations']=False
new=run('threaded',c,J/'candidate/src')
old=[[json.loads(x) for x in (J/f'baseline/rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
before=timing(old);after=timing(new)
lossdiff=max(abs(x['loss']-y['loss']) for a,b in zip(old,new) for x,y in zip(a,b))
a=torch.load(J/'baseline/last.pt',map_location='cpu',weights_only=False)
b=torch.load(J/'threaded/last.pt',map_location='cpu',weights_only=False)
modeldiff=max((a['model'][k]-b['model'][k]).abs().max().item() for k in a['model'])
optdiff=max((v-b['optimizer']['state'][k][n]).abs().max().item()
            for k,s in a['optimizer']['state'].items() for n,v in s.items() if isinstance(v,torch.Tensor))
mem=[json.loads(x) for x in (J/'memory.jsonl').read_text().splitlines()];mem=[x for x in mem if x['phase']=='threaded']
speedup=before['mean_step_seconds']/after['mean_step_seconds']
passed=lossdiff<1e-5 and modeldiff<1e-5 and optdiff<1e-5 and speedup>=1.3 and max(x['usage']/x['limit'] for x in mem)<.75
r=dict(passed=passed,baseline=before,optimized=after,speedup=speedup,loss_max_abs_difference=lossdiff,
       model_max_abs_difference=modeldiff,optimizer_max_abs_difference=optdiff,
       scheduler_equal=a['scheduler']==b['scheduler'],sampler_equal=a['sampler_position']==b['sampler_position']==21530,
       config_sha256=hashlib.sha256((J/'threaded.yaml').read_bytes()).hexdigest(),
       peak_container_bytes=max(x['usage'] for x in mem),steps_per_rank=30,world_size=8,
       selected_config='threaded.yaml',selected_checkpoint='threaded/last.pt')
(J/'threaded_comparison.json').write_text(json.dumps(r,indent=2));status('ready_for_review' if passed else 'threaded_failed',comparison=r)
