"""Collect missing train visibility and audit reused values on eight shared GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve();root=Path(__file__).resolve().parents[1]
    m=json.loads((out/'manifest.json').read_text());assert m['prepared'] and m['split']=='train'
    (out/'runner.lock').open('x').write(str(os.getpid()));jobs=[];status=dict(phase='collecting',started=time.time(),pid=os.getpid(),jobs=[],resource_scope='Lightweight annotation/render workers share active training GPUs; no training code or schedule changes, no timing benchmark claim')
    def save():
        temp=out/'status.tmp';temp.write_text(json.dumps(dict(status,updated=time.time()),indent=2));temp.replace(out/'status.json')
    common=['--runtime',m['runtime'],'--data-root',m['data_root'],'--index-root',m['index_root'],'--out',str(out),'--world',str(m['world'])]
    try:
        for rank in range(m['world']):
            command=[sys.executable,str(root/'tools/collect_natural_visibility.py'),'worker',*common,'--rank',str(rank)]
            log=(out/f'rank{rank}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            proc=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);jobs.append((proc,log));status['jobs'].append(dict(rank=rank,pid=proc.pid,command=command))
        save()
        while any(proc.poll() is None for proc,log in jobs):save();time.sleep(5)
        failures=[]
        for rank,(proc,log) in enumerate(jobs):
            log.close()
            if proc.returncode:failures.append(dict(rank=rank,returncode=proc.returncode))
        if failures:raise RuntimeError('Raw checks or collection failed: '+json.dumps(failures))
        status['phase']='merging';save()
        with (out/'merge.log').open('w') as log:subprocess.run([sys.executable,str(root/'tools/collect_natural_visibility.py'),'merge',*common],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
        status['phase']='completed';save()
    except BaseException as error:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,log in jobs:proc.wait();log.close()
        status.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
