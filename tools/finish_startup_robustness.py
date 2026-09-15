"""Read-only robustness analysis after the already running startup factorial finishes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path)
    p.add_argument('--controller-pid',required=True,type=int);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=False)
    status=dict(pid=os.getpid(),controller_pid=a.controller_pid,experiment=str(a.experiment),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save(phase,**extra):
        status.update(phase=phase,updated=time.time(),**extra);tmp=a.out/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(a.out/'status.json')
    try:
        while True:
            s=json.loads((a.experiment/'status.json').read_text())
            if s['phase']=='completed':break
            if s['phase']=='failed':raise RuntimeError('Training/evaluation failed: '+str(s.get('error')))
            proc=Path('/proc')/str(a.controller_pid)
            if not proc.exists():raise RuntimeError('Controller handle missing before terminal result')
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'run_startup_factorial.py' not in command or str(a.experiment) not in command:raise RuntimeError('Controller identity changed')
            save('waiting',controller_phase=s['phase']);time.sleep(15)
        e=json.loads((a.experiment/'experiment.json').read_text());comparison=json.loads((a.experiment/'comparison.json').read_text());assert comparison['completed']
        folders={'parent':e['reference_evaluation'],'previous_candidate':e['previous_candidate']['evaluation'],
            **{arm:str(a.experiment/arm/'s0_val') for arm in e['arms']}}
        flags=[v for name,path in folders.items() for v in ('--evaluation',name+'='+path)]
        cmd=[sys.executable,str(root/'tools/analyze_pose_robustness.py'),*flags,'--reference','previous_candidate','--out',str(a.out/'analysis')]
        save('analyzing',command=cmd)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (a.out/'analysis.log').open('w') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        save('completed',report=str(a.out/'analysis/report.md'))
    except BaseException as error:
        save('failed',error=repr(error));raise


if __name__=='__main__':main()
