"""Evaluate one saved streaming checkpoint on the complete s0 validation population."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from lip.evaluate_stream import merge_shards


def training_progress(path):
    if path is None or not path.is_file():return None
    rows=[]
    for line in path.read_text().splitlines()[-50:]:
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    if not rows:return None
    return dict(step=rows[-1]['new_stage_step'],mean_recent_seconds=statistics.mean(r['seconds'] for r in rows))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--index-root',default='cache/dexycb_s0')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--share-gpus',action='store_true')
    p.add_argument('--training-log',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    if active and not a.share_gpus:raise RuntimeError('Active GPU jobs; sharing must be explicitly requested')
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    devices=[tuple(map(int,line.split(','))) for line in memory.strip().splitlines()]
    if len(devices)!=8 or any(free<4096 for _,free in devices):raise RuntimeError('Expected eight GPUs with at least 4 GiB free each')
    streams=[json.loads(line) for line in (Path(a.index_root)/'streams.jsonl').read_text().splitlines()]
    streams=sorted((s for s in streams if s['split']=='val'),key=lambda s:s['stream_id'])
    assert len(streams)==320 and sum(s['num_frames'] for s in streams)==23200
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(a.checkpoint).resolve();config=Path(a.config).resolve()
    receipt=dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checkpoint=str(checkpoint),
                 checkpoint_sha256=sha(checkpoint),source_sha256=source_hash(),gpu_sharing=a.share_gpus,
                 existing_compute_processes=active,free_memory_mib=devices,training_before=training_progress(a.training_log),
                 latency_benchmark=False,expected_streams=320,expected_frames=23200,commands=[])
    jobs=[]
    try:
        for rank,(gpu,_) in enumerate(devices):
            command=[sys.executable,'-m','lip.evaluate_stream','--config',str(config),'--checkpoint',str(checkpoint),
                     '--data-root',str(Path(a.data_root).resolve()),'--index-root',str(Path(a.index_root).resolve()),
                     '--out',str(a.out/f'rank{rank}'),'--rank',str(rank),'--world-size','8','--split','val']
            env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            log=(a.out/f'rank{rank}.log').open('w')
            proc=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log));receipt['commands'].append(dict(command=command,gpu=gpu,pid=proc.pid))
        (a.out/'launch.json').write_text(json.dumps(receipt,indent=2))
        while any(proc.poll() is None for proc,log in jobs):
            if any(proc.poll() not in (None,0) for proc,log in jobs):raise RuntimeError('Evaluation shard failed; inspect its log')
            progress=[]
            for rank in range(8):
                f=a.out/f'rank{rank}/predictions.jsonl'
                progress.append(sum(1 for _ in f.open()) if f.exists() else 0)
            (a.out/'status.json').write_text(json.dumps(dict(phase='evaluating',frames_per_rank=progress,
                completed_frames=sum(progress),training=training_progress(a.training_log),utc=time.time()),indent=2))
            time.sleep(5)
        for proc,log in jobs:assert proc.returncode==0;log.close()
        merge_shards(a.out,8)
        manifest=json.loads((a.out/'manifest.json').read_text())
        assert manifest['population_verified'] and manifest['frames']==23200 and len(manifest['streams'])==320
        assert manifest['checkpoint_sha256']==receipt['checkpoint_sha256'] and manifest['source_sha256']==receipt['source_sha256']
        assert manifest['fp_calls']==manifest['critic_calls']==0 and not manifest['depth_correction']
        receipt.update(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),training_after=training_progress(a.training_log))
        (a.out/'launch.json').write_text(json.dumps(receipt,indent=2))
        (a.out/'status.json').write_text(json.dumps(dict(phase='completed',frames=23200,streams=320,checkpoint_sha256=receipt['checkpoint_sha256']),indent=2))
        print(json.dumps(dict(completed=True,frames=23200,streams=320,out=str(a.out)),indent=2),flush=True)
    except BaseException as error:
        # Only this launcher's evaluation children can be stopped. Existing
        # training processes are never signaled or reconfigured.
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,log in jobs:proc.wait();log.close()
        (a.out/'status.json').write_text(json.dumps(dict(phase='failed',error=repr(error)),indent=2))
        raise


if __name__=='__main__':main()
