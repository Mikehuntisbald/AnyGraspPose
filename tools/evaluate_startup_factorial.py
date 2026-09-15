"""Run all four fixed step-1000 checkpoints with one supplied initializer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.evaluate_stream import merge_shards
from lip.engine.stream_checkpoint import sha
from compare_startup_factorial import ARMS,compare


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--initial-poses',required=True,type=Path);a=p.parse_args()
    a.out=a.out.resolve();a.initial_poses=a.initial_poses.resolve();e=json.loads((a.out/'experiment.json').read_text())
    if not a.initial_poses.is_file():raise FileNotFoundError(a.initial_poses)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('Evaluation requires the 8 GPUs to be idle')
    root=Path(__file__).resolve().parents[1];jobs=[];status=dict(phase='evaluating',initial_poses_sha256=sha(a.initial_poses),commands=[])
    try:
        for ai,arm in enumerate(ARMS):
            folder=a.out/arm;receipt=json.loads((folder/'training_receipt.json').read_text())
            assert receipt['completed'] and receipt['step']==1000 and receipt['checkpoint_sha256']==sha(folder/'train/last.pt')
            (folder/'s0_val').mkdir(exist_ok=False)
            for rank in range(2):
                gpu=2*ai+rank;env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                command=[sys.executable,'-m','lip.evaluate_stream','--config',e['arms'][arm]['config'],'--checkpoint',str(folder/'train/last.pt'),
                    '--out',str(folder/f's0_val/rank{rank}'),'--data-root',e['data_root'],'--index-root',e['index_root'],
                    '--split','val','--rank',str(rank),'--world-size','2','--initial-poses',str(a.initial_poses)]
                log=(folder/f's0_val/rank{rank}.log').open('w');proc=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                jobs.append((proc,log));status['commands'].append(dict(arm=arm,gpu=gpu,pid=proc.pid,command=command))
        while any(p.poll() is None for p,l in jobs):
            if any(p.poll() not in (None,0) for p,l in jobs):raise RuntimeError('Evaluation shard failed')
            status['updated']=time.time();(a.out/'evaluation_status.json').write_text(json.dumps(status,indent=2));time.sleep(5)
        for proc,log in jobs:assert proc.returncode==0;log.close()
        for arm in ARMS:merge_shards(a.out/arm/'s0_val',2)
        compare(a.out);status['phase']='completed'
    except BaseException as error:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,log in jobs:proc.wait();log.close()
        status.update(phase='failed',error=repr(error));raise
    finally:(a.out/'evaluation_status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
