"""Bind the matched-run result, complete test suite and preserved resume state."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import yaml

J=Path('runs/basin_speed_v1')
r=json.loads((J/'threaded_comparison.json').read_text())
# The original trajectory-bitwise diagnostic remains failed and preserved.
# Acceptance here is separately scoped to functional equivalence and throughput.
assert json.loads((J/'input_audit.json').read_text())['passed']
assert json.loads((J/'loader.json').read_text())['resume_order_equal']
assert json.loads((J/'gpu_equivalence.json').read_text())['passed']
assert r['scheduler_equal'] and r['sampler_equal'] and r['speedup']>=1.3
assert r['peak_container_bytes'] < .75 * 512 * 2**30
root=ET.parse(J/'tests_gpu.xml').getroot()
suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
assert counts['tests']>=45 and counts['failures']==counts['errors']==0 and counts['skipped']<=1
old=yaml.safe_load((J/'baseline.yaml').read_text());new=yaml.safe_load((J/'threaded.yaml').read_text())
allowed={'run_id','decode_threads_per_worker','preload_rollout_observations'}
assert {k:v for k,v in old.items() if k not in allowed}=={k:v for k,v in new.items() if k not in allowed}
assert new['decode_threads_per_worker']==4 and not new['preload_rollout_observations']
assert new['num_workers_per_rank']==new['prefetch_factor']==1 and not new['pin_memory']
assert new['basin_quality_policy']=='validated' and new['fpaware_enabled']
ck=torch.load(J/'threaded/last.pt',map_location='cpu',weights_only=False)
assert ck['global_step']==ck['scheduler']['last_epoch']==ck['sampler_position']==21530
source=J/'candidate/src/lip';h=hashlib.sha256()
for f in sorted(source.rglob('*.py')):
    h.update(str(f.relative_to(source)).encode());h.update(f.read_bytes())
assert h.hexdigest()==ck['code_sha256']
cfg_hash=hashlib.sha256((J/'threaded.yaml').read_bytes()).hexdigest();assert cfg_hash==r['config_sha256']
reference=torch.load(J/'baseline/last.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(a['torch'],b['torch']) and torch.equal(a['cuda'],b['cuda'])
           for a,b in zip(reference['rng'],ck['rng']))
for rank in range(8):
    rows=[json.loads(x) for x in (J/f'threaded/rank{rank}.jsonl').read_text().splitlines()]
    assert len(rows)==30 and all(x['nonfinite_count']==0 and x['step']==x['scheduler_step']==x['sampler_position'] for x in rows)
    assert all(x['basin_weight']==.01 and abs(x['loss']-x['pose_loss']-.01*x['basin_loss'])<2e-6 for x in rows)
receipt=dict(passed=True,config_sha256=cfg_hash,code_sha256=h.hexdigest(),tests=counts,
             acceptance_scope='functional equivalence, state restoration, finite DDP updates, bounded memory and throughput; not bitwise multi-step trajectories',
             trajectory_bitwise_equal=False,
             trajectory_diagnostic='threaded_comparison.json',baseline_repeatability='baseline_repeatability.json',
             limitation='Specific source of initial ~1e-6 DDP gradient-norm discrepancy not isolated; it amplifies over BF16 closed-loop updates. Baseline 8-step repeat was exact. No accuracy-equivalence claim.',
             comparison_sha256=hashlib.sha256((J/'threaded_comparison.json').read_bytes()).hexdigest(),
             resume=str(J/'threaded/last.pt'),resume_step=21530,
             resume_sha256=hashlib.sha256((J/'threaded/last.pt').read_bytes()).hexdigest(),
             speedup=r['speedup'],only_config_changes=sorted(allowed),rng_states_equal=True,
             input_audit_sha256=hashlib.sha256((J/'input_audit.json').read_bytes()).hexdigest(),
             gpu_equivalence_sha256=hashlib.sha256((J/'gpu_equivalence.json').read_bytes()).hexdigest())
(J/'approval.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
