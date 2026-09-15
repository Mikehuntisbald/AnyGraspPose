"""Collect frozen proposals on 8 GPUs, then train and audit the quality head."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','checkpoint','data-root','index-root','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--reference-cache',type=Path);p.add_argument('--full-streams',action='store_true');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];os.chdir(root);a.out=a.out.resolve()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('Collection requires idle GPUs')
    if a.workers<1 or a.workers%8:raise ValueError('Use a positive multiple of 8 workers')
    base=[sys.executable,'tools/collect_update_quality_full.py' if a.full_streams else 'tools/collect_update_quality.py']
    for key in ('config','checkpoint','data-root','index-root','out'):base+=['--'+key,str(getattr(a,key.replace('-','_')).resolve())]
    if a.full_streams:
        if a.reference_cache is None:raise ValueError('Full streams preserve an explicit existing physical partition')
        base+=['--reference-cache',str(a.reference_cache.resolve())]
    base+=['--world',str(a.workers)]
    subprocess.run([*base,'--prepare'],check=True)
    status=dict(phase='collecting',commands=[],started=time.time());jobs=[]
    def save():(a.out/'status.json').write_text(json.dumps(status,indent=2))
    try:
        for rank in range(a.workers):
            env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=str(rank%8),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            command=[*base,'--rank',str(rank)];log=(a.out/f'collect_rank{rank}.log').open('w')
            proc=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);jobs.append((proc,log))
            status['commands'].append(dict(rank=rank,pid=proc.pid,command=command))
        save()
        while any(p.poll() is None for p,l in jobs):
            if any(p.poll() not in (None,0) for p,l in jobs):raise RuntimeError('A collection worker failed')
            status['updated']=time.time();save();time.sleep(5)
        for proc,log in jobs:log.close();assert proc.returncode==0
        jobs=[];status['phase']='training_quality';save()
        command=[sys.executable,'tools/train_update_quality.py','--cache',str(a.out),'--out',str(a.out/'fit'),'--steps','2000']
        env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (a.out/'fit.log').open('w') as log:subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((a.out/'fit/result.json').read_text());assert result['completed'];status.update(phase='completed',result=result,completed=time.time());save()
    except BaseException as error:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,log in jobs:proc.wait();log.close()
        status.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
