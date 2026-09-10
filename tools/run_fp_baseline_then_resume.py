"""Checkpoint-boundary FP evaluation transaction; always restart training after eval."""
import json,os,signal,subprocess,time,shutil
from pathlib import Path
ROOT=Path('/mnt/why/dexycb_lip');os.chdir(ROOT)
JOB=ROOT/'runs/fp_baseline_20260910';JOB.mkdir(exist_ok=True,parents=True)
TRAIN=ROOT/'runs/lip_v1_s0';STATUS=ROOT/'runs/full_train_after_upload_20260910/status.json'
def status(phase,**kw):
 r=dict(phase=phase,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
 p=JOB/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p);print(json.dumps(r),flush=True)
def latest():return json.loads((TRAIN/'rank0.jsonl').read_text().splitlines()[-1])['step']
# Require a newly atomically committed checkpoint; preserve it before interruption.
initial=(TRAIN/'last.pt').stat().st_mtime_ns
status('waiting_new_checkpoint',step=latest())
while (TRAIN/'last.pt').stat().st_mtime_ns==initial:time.sleep(.2)
flow=json.loads(STATUS.read_text());controller=flow['pid'];group=flow['child_pid']
assert flow['phase']=='training' and os.getpgid(group)==group
assert b'auto_full_train.py' in Path(f'/proc/{controller}/cmdline').read_bytes()
assert b'torch.distributed.run' in Path(f'/proc/{group}/cmdline').read_bytes()
shutil.copy2(TRAIN/'last.pt',JOB/'resume.pt')
# Stop the existing orchestration and its training group; it has a SIGTERM cleanup handler.
os.kill(controller,signal.SIGTERM)
code=1
try:
 for _ in range(120):
  r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
  if not r.stdout.strip():break
  time.sleep(1)
 else:raise RuntimeError('GPUs not released; inspect before running baseline')
 import torch
 ck=torch.load(JOB/'resume.pt',map_location='cpu',weights_only=False)
 status('evaluating',checkpoint_step=ck['global_step'],last_logged_step=latest(),replayed_steps=latest()-ck['global_step'])
 del ck
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',MAX_JOBS='4',TMPDIR=str(ROOT/'cache/tmp'))
 cmd=[str(ROOT/'.venv-fp/bin/python'),'-u','tools/eval_foundationpose.py','--fp-root','third_party/FoundationPose','--data-root',str(ROOT/'cache/raw_full_20260910'),'--reference','runs/lip_v1_s0/val_10000','--out',str(JOB/'eval')]
 with (JOB/'eval.log').open('w') as f:code=subprocess.call(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 if code==0:
  with (JOB/'comparison.log').open('w') as f:code=subprocess.call([str(ROOT/'.venv-fp/bin/python'),'tools/compare_foundationpose.py','--lip','runs/lip_v1_s0/val_10000','--fp',str(JOB/'eval'),'--out',str(JOB/'comparison.json')],env=env,stdout=f,stderr=subprocess.STDOUT)
finally:
 status('resuming_training',eval_exit_code=code)
 env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env['DEX_YCB_DIR']=str(ROOT/'cache/raw_full_20260910')
 with (JOB/'training_resumed.log').open('a') as f:
  child=subprocess.Popen(['bash','scripts/train_8gpu.sh','--resume',str(JOB/'resume.pt')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  status('training_resumed',child_pid=child.pid,eval_exit_code=code)
  rc=child.wait()
 status('training_exited',returncode=rc,eval_exit_code=code)
 if rc==0:
  subprocess.check_call([str(ROOT/'.venv/bin/python'),'tools/verify_training_completion.py','--expected-steps','40000'])
