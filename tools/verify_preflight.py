import argparse,hashlib,json
from pathlib import Path
import yaml
import torch
from lip.engine.config import check_data_gate
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--approve',action='store_true');a=p.parse_args()
path=Path(a.config);c=yaml.safe_load(path.read_text());audit=check_data_gate('cache/dexycb_s0')
report=json.loads(Path('runs/overfit32/report.json').read_text())
if report.get('source')!='real_DexYCB' or not report.get('passed') or report['num_clips']!=32 or report.get('steps')!=500 or not report.get('data_complete'):raise RuntimeError('Full-data real 32-clip overfit has not passed')
overfit_manifest=json.loads(Path('runs/overfit32/manifest.json').read_text())
if overfit_manifest['split_hash']!=audit['split_hash']:raise RuntimeError('Stale overfit manifest')
probe=json.loads(path.with_suffix('.probe.json').read_text())
if probe['source']!='real_DexYCB' or not probe.get('data_complete'):raise RuntimeError('Full-data batch probe missing')
if c['split_hash']!=audit['split_hash'] or c['mesh_hash']!=audit['mesh_hash']:raise RuntimeError('Stale resolved config')
if c['batch_size_per_gpu'] not in probe['eligible_batches'] or c['batch_size_per_gpu']*c['grad_accum_steps']*8!=256:raise RuntimeError('Unsafe batch resolution')
rank_steps=[]
for rank in range(8):
 rows=[json.loads(x) for x in Path(f'runs/ddp_real/rank{rank}.jsonl').read_text().splitlines()]
 bystep={r['step']:r for r in rows}
 if any(s not in bystep for s in range(1,54)):raise RuntimeError('Missing DDP/resume steps')
 if any(r['nonfinite_count'] or r['scheduler_step']!=r['step'] for r in rows):raise RuntimeError('Invalid DDP step')
 rank_steps.append(bystep[53]['sampler_position'])
if len(set(rank_steps))!=1:raise RuntimeError('DDP sampler divergence')
checkpoint=torch.load('runs/ddp_real/last.pt',map_location='cpu',weights_only=False)
if checkpoint['split_hash']!=audit['split_hash'] or checkpoint['mesh_hash']!=audit['mesh_hash']:raise RuntimeError('Stale DDP checkpoint')
if checkpoint['global_step']!=53 or checkpoint['scheduler']['last_epoch']!=53 or len(checkpoint['rng'])!=8:raise RuntimeError('Invalid DDP resume checkpoint')
log=Path('runs/pytest_preflight.log').read_text()
if ' passed' not in log or 'failed' in log or 'ERROR' in log:raise RuntimeError('Required pytest receipt missing')
if not Path('runs/ddp_real_resume.log').exists() or 'resumed_step' not in Path('runs/ddp_real_resume.log').read_text():raise RuntimeError('Resume receipt missing')
if a.approve:
 c.update(preflight_approved=True,official_s0_complete=True,engineering_preflight_passed=True,preflight_scope='full official s0, all 10 subjects')
 path.write_text(yaml.safe_dump(c,sort_keys=False))
 Path('runs/preflight_passed.json').write_text(json.dumps(dict(passed=True,split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],rank_sampler_positions=rank_steps,config_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
else:
 receipt=json.loads(Path('runs/preflight_passed.json').read_text())
 if not c.get('preflight_approved') or receipt['config_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise RuntimeError('Preflight approval missing/stale')
print('REAL DATA PREFLIGHT PASSED; no long training was launched by this check')
