"""Supplement held-out dense correspondence metrics; no extra training."""
import json,os,subprocess,sys,time,hashlib
from pathlib import Path

runtime=Path(__file__).resolve().parents[1]
parent=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/dense_canonical_v74')
assert json.loads((parent/'status.json').read_text())['stage']=='complete'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
root=parent/'dense_audit';root.mkdir(exist_ok=False)
files={str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('src','tools','configs')
       for p in (runtime/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(root/'source_receipt.json').write_text(json.dumps(files,indent=2)+'\n')
for angle,records in ((0,4),(10,8),(60,4)):
 children=[];logs=[]
 try:
  for rank in range(8):
   log=(root/f'angle{angle}.rank{rank}.log').open('w');logs.append(log)
   env=dict(os.environ,PYTHONPATH=str(runtime/'src'),CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
   command=[sys.executable,'tools/probe_dense_canonical_v74.py','--config','configs/jepa/dense_canonical_v74_surface.yaml',
       '--checkpoint',str(parent/'surface/seed42/last.pt'),'--out',str(root/f'angle{angle}'/f'rank{rank}'),
       '--rank',str(rank),'--world','8','--records',str(records),'--seed-start','74050000','--rotation-degrees',str(angle)]
   children.append(subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT))
  while any(p.poll() is None for p in children):
   if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Dense audit worker failed')
   (root/'status.json').write_text(json.dumps(dict(stage='running',angle=angle,pids=[p.pid for p in children])))
   time.sleep(3)
  assert all(p.returncode==0 for p in children)
 finally:
  for p in children:
   if p.poll() is None:p.terminate()
  for f in logs:f.close()
 for rank in range(8):
  original=[json.loads(s) for s in (parent/'probe'/f'surface_{angle}'/f'rank{rank}'/'frames.jsonl').read_text().splitlines()]
  audit=[json.loads(s) for s in (root/f'angle{angle}'/f'rank{rank}'/'frames.jsonl').read_text().splitlines()]
  assert len(original)==len(audit)==records
  for a,b in zip(original,audit):
   for key in ('seed','stream','heavy','natural','metrics','flow'):assert a[key]==b[key],(angle,rank,key)
(root/'status.json').write_text(json.dumps(dict(stage='complete',frames=128,primary_records_exact=True)))
