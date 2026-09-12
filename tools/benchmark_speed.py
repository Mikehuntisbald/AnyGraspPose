"""Matched 8-GPU benchmark; never launches production before review of its receipt."""
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import torch
import yaml

R=Path('/mnt/why/dexycb_lip');os.chdir(R);J=R/'runs/basin_speed_v1'

def status(phase,**kw):
    (J/'status.json').write_text(json.dumps(dict(phase=phase,utc=time.time(),**kw),indent=2))

def memory(phase):
    p=Path('/sys/fs/cgroup/memory');s=dict(x.split() for x in (p/'memory.stat').read_text().splitlines())
    row=dict(utc=time.time(),phase=phase,usage=int((p/'memory.usage_in_bytes').read_text()),
             limit=int((p/'memory.limit_in_bytes').read_text()),rss=int(s['rss']),shmem=int(s['shmem']))
    with (J/'memory.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return row

def env(source):
    e=os.environ.copy();e.update(PYTHONPATH=str(source),TORCH_HOME=str(R/'cache/torch'),
      TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),TMPDIR=str(R/'cache/tmp'),
      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    return e

def run(kind,c,source,steps=30):
    cfg=J/f'{kind}.yaml';cfg.write_text(yaml.safe_dump(c,sort_keys=False))
    cmd=[str(R/'.venv-fp/bin/python'),'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
         '-m','lip.train','--config',str(cfg),'--data-root',str(R/'cache/raw_full_20260910'),
         '--index-root','cache/dexycb_s0','--output',str(J/kind),'--resume',str(J/'resume_21500.pt'),
         '--max-steps',str(21500+steps)]
    with (J/f'{kind}.log').open('w') as f:
        p=subprocess.Popen(cmd,env=env(source),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        status(kind,child_pid=p.pid)
        while p.poll() is None:memory(kind);time.sleep(5)
        assert p.returncode==0, f'{kind} failed: {p.returncode}'
    rows=[]
    for rank in range(8):
        rr=[json.loads(x) for x in (J/kind/f'rank{rank}.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rr]==list(range(21501,21501+steps))
        assert {r['history_mode'] for r in rr}=={'noisy_gt','lip_only','lip_fp'}
        assert all(r['step']==r['scheduler_step']==r['sampler_position'] and r['nonfinite_count']==0
                   and r['basin_weight']==.01 and r['effective_batch']==256 for r in rr)
        assert all(abs(r['loss']-r['pose_loss']-.01*r['basin_loss'])<2e-6 for r in rr)
        rows.append(rr)
    return rows

def timing(rows):
    # Paired sample indices; drop the first eight steps (startup/model warm-up).
    steps=[dict(step=rows[0][i]['step'],mode=rows[0][i]['history_mode'],
                seconds=max(rank[i]['elapsed'] for rank in rows)) for i in range(8,30)]
    return dict(mean_step_seconds=statistics.mean(x['seconds'] for x in steps),
                median_step_seconds=statistics.median(x['seconds'] for x in steps),
                modes={m:dict(steps=sum(x['mode']==m for x in steps),
                              mean_seconds=statistics.mean(x['seconds'] for x in steps if x['mode']==m))
                       for m in ['noisy_gt','lip_only','lip_fp']},
                peak_gpu_bytes=max(x['peak_memory_bytes'] for rr in rows for x in rr))

def main():
    status('waiting_checkpoint_21500')
    while not (J/'paused.json').exists():memory('waiting');time.sleep(2)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(2)
    assert json.loads((J/'loader.json').read_text())['passed']
    tests=(J/'tests.log').read_text();assert 'passed' in tests and 'failed' not in tests and 'ERROR' not in tests
    old=yaml.safe_load((R/'runs/basin_memory_fix/config.yaml').read_text());old['run_id']='speed_baseline'
    new=dict(old,run_id='basin_speed_v1',decode_threads_per_worker=4,preload_rollout_observations=True)
    baseline=run('baseline',old,R/'runs/basin_memory_fix/candidate/src')
    optimized=run('optimized',new,J/'candidate/src')
    before=timing(baseline);after=timing(optimized)
    loss_difference=max(abs(a['loss']-b['loss']) for ar,br in zip(baseline,optimized) for a,b in zip(ar,br))
    ca=torch.load(J/'baseline/last.pt',map_location='cpu',weights_only=False)
    cb=torch.load(J/'optimized/last.pt',map_location='cpu',weights_only=False)
    model_difference=max((ca['model'][k]-cb['model'][k]).abs().max().item() for k in ca['model'])
    optimizer_difference=max((va-vb).abs().max().item()
        for k,sa in ca['optimizer']['state'].items()
        for n,va in sa.items() if isinstance(va,torch.Tensor)
        for vb in [cb['optimizer']['state'][k][n]])
    assert ca['global_step']==cb['global_step']==21530
    assert ca['scheduler']==cb['scheduler']
    assert ca['sampler_position']==cb['sampler_position']==21530
    mem=[json.loads(x) for x in (J/'memory.jsonl').read_text().splitlines()]
    mem=[x for x in mem if x['phase']=='optimized']
    speedup=before['mean_step_seconds']/after['mean_step_seconds']
    passed=loss_difference<1e-5 and model_difference<1e-5 and optimizer_difference<1e-5 and speedup>=1.3 and max(x['usage']/x['limit'] for x in mem)<.75
    receipt=dict(passed=passed,baseline=before,optimized=after,speedup=speedup,loss_max_abs_difference=loss_difference,
                 model_max_abs_difference=model_difference,optimizer_max_abs_difference=optimizer_difference,
                 scheduler_equal=True,sampler_equal=True,steps_per_rank=30,world_size=8,
                 config_sha256=hashlib.sha256((J/'optimized.yaml').read_bytes()).hexdigest(),
                 peak_container_bytes=max(x['usage'] for x in mem),
                 source_checkpoint_sha256=hashlib.sha256((J/'resume_21500.pt').read_bytes()).hexdigest())
    (J/'comparison.json').write_text(json.dumps(receipt,indent=2));status('ready_for_review' if passed else 'benchmark_failed',comparison=receipt)

if __name__=='__main__':main()
