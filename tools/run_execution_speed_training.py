"""Resume a verified execution-only candidate and finish the existing25000 budget."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.jepa_checkpoint import sha
from lip.unified.checkpoint import atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];c=yaml.safe_load(Path(a.config).read_text());out=Path(c['paths']['output']);source=c['performance_resume']
 bench=json.loads((root/'benchmark.json').read_text());assert bench['passed'] and bench['persisted_updates']==0 and bench['speedup']>1.05
 assert bench['checkpoint_sha256']==source['sha256'] and c['training']['max_steps']==25000
 flags=bench['arms'][bench['best']]['flags'];assert all(c['runtime'][k]==v for k,v in flags.items())
 directory=root/'controller';directory.mkdir(exist_ok=True);lock=(directory/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 files={str(f.relative_to(root)):sha(f) for folder in ('src','tools','configs','tests') for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
 atomic_json(root/'runtime_receipt.json',dict(files=files,benchmark_sha256=sha(root/'benchmark.json')))
 state=dict(completed=False,pid=os.getpid(),source_step=source['step'],target_step=25000,mode='execution_only_speed_migration')
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
  args+=['--resume',str(out/'last.pt')] if (out/'last.pt').exists() else ['--performance-from',source['checkpoint']]
  run('train_to_'+str(step),args)
  receipt=json.loads((out/'last.receipt.json').read_text());assert receipt['step']==step and receipt['all_rank_rng']==8
 try:
  if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
  train(source['step']+2);train(source['step']+3)
  run('startup_audit',[sys.executable,'tools/verify_execution_resume.py','--config',a.config,'--out',str(root/'startup_receipt.json')])
  train(25000)
  run('final_audit',[sys.executable,'tools/verify_execution_resume.py','--config',a.config,'--out',str(root/'final_state_receipt.json')])
  run('evaluate25000',[sys.executable,'tools/evaluate_recovery_focus.py','--config',a.config,'--checkpoint',str(out/'last.pt'),'--out',str(root/'diagnostics/step25000')])
  run('report',[sys.executable,'tools/report_recovery_extension.py','--root',str(root)])
  status('complete',completed=True,checkpoint_sha256=sha(out/'last.pt'))
 except Exception as e:
  if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
  status('failed',error=str(e));raise

if __name__=='__main__':main()
