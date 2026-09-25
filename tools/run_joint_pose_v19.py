"""Bounded 35400->45400 joint training with serialized native-val and recovery evaluation."""
import argparse,json,os,signal,subprocess,sys,time,fcntl,shutil,tarfile
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.jepa_checkpoint import sha
from lip.unified.checkpoint import atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=yaml.safe_load(Path(a.config).read_text())
 exe=Path(__file__).resolve().parents[1];out=Path(c['paths']['output']);root=out.parents[1];directory=root/'controller';directory.mkdir(exist_ok=True)
 lock=(directory/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pre=json.loads((root/'preflight/receipt.json').read_text());assert pre['passed'] and pre['config_sha256']==sha(a.config)
 assert c['training']['max_steps']-c['migration']['source_step']==10000
 files={str(f.relative_to(exe)):sha(f) for folder in ('src','tools','configs','tests') for f in (exe/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
 atomic_json(root/'runtime_receipt.json',dict(files=files,execution_root=str(exe),artifact_root=str(root)))
 with tarfile.open(root/'source_and_config.tar.gz','w:gz') as archive:
  for f in files:archive.add(exe/f,arcname=f)
 state=dict(completed=False,pid=os.getpid(),source_step=35400,target_step=45400,mode='joint_jepa_pose');children=[]
 def status(value,**kw):state.update(status=value,heartbeat_unix=time.time(),**kw);atomic_json(directory/'status.json',state)
 def stop(sig,_):
  for child in children:
   if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
  status('interrupted');raise SystemExit(128+sig)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR='/tmp/dexycb_joint_pose_v19_inductor',TRITON_CACHE_DIR='/tmp/dexycb_staged_rope_v18_triton')
 def run(name,commands,sharded=False):
  nonlocal children
  assert all(sha(exe/f)==h for f,h in files.items()),'Pinned source changed'
  children=[];logs=[]
  try:
   for rank,command in enumerate(commands):
    log=(directory/(name+f'.{rank}.log')).open('a');logs.append(log)
    e=dict(env,CUDA_VISIBLE_DEVICES=str(rank)) if sharded else env
    children.append(subprocess.Popen([sys.executable]+command,cwd=exe,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
   while any(child.poll() is None for child in children):
    if any(child.poll() not in (None,0) for child in children):raise RuntimeError(name+' failed')
    status('running',job=name,child_pids=[x.pid for x in children]);time.sleep(5)
   if any(child.returncode for child in children):raise RuntimeError(name+' failed')
  finally:
   for child in children:
    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
   for log in logs:log.close()
 def train(step):
  args=['-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_two_stream.py','--config',a.config,'--stop-at',str(step)]
  if (out/'last.pt').exists():args+=['--resume',str(out/'last.pt')]
  run('train_to'+str(step),[args]);assert json.loads((out/'last.receipt.json').read_text())['step']==step
 def audit(name):run(name,[['tools/audit_joint_pose.py','--config',a.config,'--out',str(root/(name+'.json'))]])
 def native(name,checkpoint):
  pred=root/'validation'/name;score=root/'validation'/(name+'_scored')
  run('native_'+name,[['tools/infer_unified_jepa_val.py','--config',a.config,'--checkpoint',str(checkpoint),'--out',str(pred/f'rank{rank}'),'--rank',str(rank),'--world','8','--disable-history'] for rank in range(8)],True)
  run('score_'+name,[['tools/score_unified_jepa_val.py','--run',str(pred),'--world','8','--index-root',c['paths']['index_root'],'--visibility-reference',c['paths']['visibility_reference'],'--workers','16','--out',str(score)]])
  m=json.loads((score/'manifest.json').read_text());assert m['completed'] and m['frames']==23200 and m['checkpoint_sha256']==sha(checkpoint)
 try:
  if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
  train(35402);train(35403);audit('startup_audit')
  native('source35400',out/'initial.pt')
  for step in (36400,40400,45400):
   train(step);audit('audit'+str(step));(root/'milestones').mkdir(exist_ok=True);checkpoint=root/'milestones'/f'step{step}.pt';shutil.copy2(out/'last.pt',checkpoint)
   native('step'+str(step),checkpoint)
   run('recovery'+str(step),[['tools/evaluate_recovery_focus.py','--config',a.config,'--checkpoint',str(checkpoint),'--out',str(root/'diagnostics'/f'step{step}')]])
  status('complete',completed=True,checkpoint_sha256=sha(out/'last.pt'))
 except Exception as error:status('failed',error=str(error));raise
if __name__=='__main__':main()
