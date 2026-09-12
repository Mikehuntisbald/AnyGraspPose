"""Launch the reviewed speed configuration and retain live memory evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--resume',required=True);a=p.parse_args()
R=Path('/mnt/why/dexycb_lip');os.chdir(R);J=R/'runs/basin_speed_v1'
assert json.loads((J/'approval.json').read_text())['passed']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPUs are in use'
env=os.environ.copy();env['DEX_YCB_DIR']=str(R/'cache/raw_full_20260910')
cmd=['bash','scripts/train_basin_fast.sh','--resume',a.resume]
def status(phase,**kw):
    (J/'status.json').write_text(json.dumps(dict(phase=phase,utc=time.time(),**kw),indent=2))
with (J/'train.log').open('a') as log:
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(phase='training',child_pid=proc.pid,resume=a.resume,command=cmd,utc=time.time(),run_id='basin_speed_v1')
    (J/'activation.json').write_text(json.dumps(receipt,indent=2))
    status('training',child_pid=proc.pid,resume=a.resume,command=cmd,run_id='basin_speed_v1')
    while proc.poll() is None:
        root=Path('/sys/fs/cgroup/memory');s=dict(x.split() for x in (root/'memory.stat').read_text().splitlines())
        row=dict(utc=time.time(),phase='training',usage=int((root/'memory.usage_in_bytes').read_text()),
                 limit=int((root/'memory.limit_in_bytes').read_text()),rss=int(s['rss']),shmem=int(s['shmem']))
        with (J/'memory.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        time.sleep(10)
    status('training_exited',returncode=proc.returncode)
    if proc.returncode==0:
        subprocess.run([str(R/'.venv-fp/bin/python'),'tools/verify_training_completion.py'],env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
