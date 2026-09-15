"""Run and score the matched native-val FP baseline in isolated GPU workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('fp-root','initializers','data-root','index-root','visibility-reference','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-weight-sha',required=True);p.add_argument('--gpus',default='0,1,2,3');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];gpus=a.gpus.split(',');a.out.mkdir(parents=True,exist_ok=False)
    files=[root/'tools'/n for n in ('infer_fp_val_non_gt.py','score_val_non_gt.py','val_non_gt_common.py','standalone_bop_state.py','streaming_bop_utils.py')]
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    state=dict(phase='starting',started=time.time(),commands=[],initializers_sha256=digest(a.initializers),tools_sha256={str(f):digest(f) for f in files});jobs=[]
    def save():
        state['updated']=time.time();f=a.out/'status.tmp';f.write_text(json.dumps(state,indent=2));f.replace(a.out/'status.json')
    env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    try:
        state['phase']='inference';save()
        for rank,gpu in enumerate(gpus):
            cmd=[sys.executable,'tools/infer_fp_val_non_gt.py','--fp-root',str(a.fp_root),'--initializers',str(a.initializers),
                '--expected-weight-sha',a.expected_weight_sha,'--data-root',str(a.data_root),'--index-root',str(a.index_root),
                '--rank',str(rank),'--world',str(len(gpus)),'--out',str(a.out/f'rank{rank}')]
            f=(a.out/f'rank{rank}.log').open('x');proc=subprocess.Popen(cmd,cwd=root,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            jobs.append((proc,f));state['commands'].append(dict(rank=rank,pid=proc.pid,command=cmd));save()
        while any(p.poll() is None for p,f in jobs):
            if any(p.poll() not in (None,0) for p,f in jobs):raise RuntimeError('FP worker failed')
            save();time.sleep(5)
        for proc,f in jobs:f.close();assert proc.returncode==0
        jobs=[];state['phase']='scoring';save()
        command=[sys.executable,'tools/score_val_non_gt.py','--run',str(a.out),'--world',str(len(gpus)),
            '--index-root',str(a.index_root),'--visibility-reference',str(a.visibility_reference),'--out',str(a.out/'scored'),'--allow-foundationpose-baseline']
        with (a.out/'score.log').open('x') as f:subprocess.run(command,cwd=root,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
        assert digest(a.initializers)==state['initializers_sha256']
        for f,h in state['tools_sha256'].items():assert digest(f)==h
        state.update(phase='completed',completed=time.time());save()
    except BaseException as error:
        for proc,f in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,f in jobs:proc.wait();f.close()
        state.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
