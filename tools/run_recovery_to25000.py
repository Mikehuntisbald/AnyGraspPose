"""Bounded JEPA-only extension, preserving Adam and EMA at step12000."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.jepa_checkpoint import sha
from lip.unified.checkpoint import atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];c=yaml.safe_load(Path(a.config).read_text());h=c['horizon_continuation'];out=Path(c['paths']['output'])
 assert h['source_step']==12000 and c['training']['max_steps']==25000 and c['budget']['updates']==13000
 old=root.parent/'local_difference_v15'
 previous=json.loads((old/'runtime_receipt.json').read_text())['files']
 for name,digest in previous.items():
  if name.startswith('src/'):assert sha(root/name)==digest,'Model/loss source changed: '+name
 assert json.loads((old/'final_state_receipt.json').read_text())['passed']
 assert sha(h['source_checkpoint'])==h['source_sha256']
 directory=root/'controller';directory.mkdir(exist_ok=True)
 lock=(directory/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 files={str(f.relative_to(root)):sha(f) for folder in ('src','tools','configs','tests') for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
 atomic_json(root/'runtime_receipt.json',dict(files=files,model_loss_identical_to=str(old),inherited_preflight=str(old/'preflight/receipt.json')))
 state=dict(completed=False,pid=os.getpid(),source_step=12000,target_step=25000,mode='jepa_only_optimizer_preserved_extension')
 child=None
 def status(value,**kw):
  state.update(status=value,heartbeat_unix=time.time(),**kw);atomic_json(directory/'status.json',state)
 def stop(sig,frame):
  if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
  status('interrupted');raise SystemExit(128+sig)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR='/tmp/dexycb_fp_v3_inductor_isolated',TRITON_CACHE_DIR='/tmp/dexycb_fp_v3_triton_isolated')
 def run(name,args):
  nonlocal child
  assert all(sha(root/f)==v for f,v in files.items()),'Pinned source changed'
  with (directory/(name+'.log')).open('a') as log:
   child=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll() is None:status('running',job=name,child_pid=child.pid);time.sleep(5)
   if child.returncode:raise RuntimeError(name+' failed; see job log')
 def train(step):
  args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_two_stream.py','--config',a.config,'--stop-at',str(step)]
  args+=['--resume',str(out/'last.pt')] if (out/'last.pt').exists() else ['--extend-from',h['source_checkpoint']]
  run('train_to_'+str(step),args)
  receipt=json.loads((out/'last.receipt.json').read_text());assert receipt['step']==step and receipt['all_rank_rng']==8
 try:
  if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
  train(12002);train(12003)
  run('startup_audit',[sys.executable,'tools/verify_recovery_extension.py','--config',a.config,'--out',str(root/'startup_receipt.json')])
  for step in (15000,20000,25000):
   train(step)
   run('audit'+str(step),[sys.executable,'tools/verify_recovery_extension.py','--config',a.config,'--out',str(root/('final_state_receipt.json' if step==25000 else f'audit{step}.json'))])
   run('evaluate'+str(step),[sys.executable,'tools/evaluate_recovery_focus.py','--config',a.config,'--checkpoint',str(out/'last.pt'),'--out',str(root/f'diagnostics/step{step}')])
   (root/'milestones').mkdir(exist_ok=True)
   if step<25000:shutil.copy2(out/'last.pt',root/f'milestones/step{step}.pt')
  run('report',[sys.executable,'tools/report_recovery_extension.py','--root',str(root)])
  status('complete',completed=True,checkpoint_sha256=sha(out/'last.pt'))
 except Exception as e:
  if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
  status('failed',error=str(e));raise

if __name__=='__main__':main()
