"""Frozen correct-reference safety probe after V60; no training/selection."""
import os,sys,json,subprocess,time
from pathlib import Path
root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/local_flow_v60_r1')
runtime=Path('/tmp/dexycb_local_flow_v60_r1')
assert json.loads((root/'status.json').read_text())['stage']=='complete'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
out=root/'zero_reference';out.mkdir(exist_ok=False)
children=[]
try:
 for arm in ('baseline','control','extra'):
  children=[]
  config='configs/jepa/flow_reconstruction_v56.yaml' if arm=='baseline' else str(root/'training'/f'{arm}.yaml')
  checkpoint='/mnt/why/dexycb_lip/unified_jepa_20260921/flow_reconstruction_v56/runs/seed42/last.pt' if arm=='baseline' else str(root/'training'/f'{arm}.pt')
  for rank in range(8):
   env=dict(os.environ,PYTHONPATH=str(runtime/'src'),CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   log=(out/f'{arm}.{rank}.log').open('w')
   children.append(subprocess.Popen([sys.executable,'tools/probe_geometry_transport.py','--config',config,'--checkpoint',checkpoint,'--out',str(out/arm/f'rank{rank}'),'--rank',str(rank),'--records','4','--seed-start','60050000','--rotation-degrees','0'],cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT))
  while any(p.poll() is None for p in children):
   if any(p.poll() not in (None,0) for p in children):raise RuntimeError(arm+' failed')
   (out/'status.json').write_text(json.dumps(dict(stage=arm,pids=[p.pid for p in children])))
   time.sleep(3)
  assert all(p.returncode==0 for p in children)
 (out/'status.json').write_text(json.dumps(dict(stage='complete',records_per_arm=32)))
finally:
 for p in children:
  if p.poll() is None:p.terminate()
