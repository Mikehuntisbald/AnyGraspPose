import os,sys,subprocess,json,time,hashlib
from pathlib import Path
root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/flow_coupling_v61')
root.mkdir(exist_ok=False)
runtime=Path('/tmp/dexycb_flow_coupling_v61_r0')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
files={str(p.relative_to(runtime)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for d in ('src','tools','configs') for p in (runtime/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(root/'source_receipt.json').write_text(json.dumps(files,indent=2))
children=[]
try:
 for rank in range(8):
  log=(root/f'rank{rank}.log').open('w')
  env=dict(os.environ,PYTHONPATH=str(runtime/'src'),CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
  children.append(subprocess.Popen([sys.executable,'tools/probe_flow_coupling_v61.py','--out',str(root/f'rank{rank}'),'--rank',str(rank),'--records','8'],cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT))
 while any(p.poll() is None for p in children):
  if any(p.poll() not in (0,None) for p in children):raise RuntimeError('worker failed')
  (root/'status.json').write_text(json.dumps(dict(stage='running',pids=[p.pid for p in children])))
  time.sleep(3)
 assert all(p.returncode==0 for p in children)
 (root/'status.json').write_text(json.dumps(dict(stage='complete',frames=64)))
except Exception as exc:
 for p in children:
  if p.poll() is None:p.terminate()
 (root/'status.json').write_text(json.dumps(dict(stage='failed',error=str(exc))))
 raise
