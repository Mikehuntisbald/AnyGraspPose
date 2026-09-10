"""Seal bounded real-data engineering evidence without approving full-s0 training."""
import json,hashlib
from pathlib import Path
import yaml
index=Path('cache/s0_verified_subset');audit=json.loads((index/'audit.json').read_text());geo=json.loads((index/'geometry_gate.json').read_text())
overfit=json.loads(Path('runs/overfit32_mixed_subset/report.json').read_text());candidate=Path('configs/candidate_8gpu_real_subset.yaml');c=yaml.safe_load(candidate.read_text());probe=json.loads(candidate.with_suffix('.probe.json').read_text())
assert audit['verified_train_subset'] and not audit['complete'] and not audit['errors']
assert geo['passed'] and overfit['passed'] and overfit['source']=='real_DexYCB' and overfit['num_clips']==32 and overfit['steps']==500
assert probe['source']=='real_DexYCB' and not probe['data_complete']
assert c['batch_size_per_gpu'] in probe['eligible_batches'] and 8*c['batch_size_per_gpu']*c['grad_accum_steps']==256
positions=[];losses=[];peaks=[];timings=[];ranks=[]
for rank in range(8):
 rows=[json.loads(x) for x in Path(f'runs/ddp_real_subset/rank{rank}.jsonl').read_text().splitlines()];by={r['step']:r for r in rows}
 assert all(step in by for step in range(1,54))
 assert all(r['step']==r['scheduler_step']==r['sampler_position'] and r['rollout']==4 and r['effective_batch']==256 and r['nonfinite_count']==0 for r in rows)
 positions.append(by[53]['samples_seen']);losses.extend(r['loss'] for r in rows);peaks.extend(r['peak_memory_bytes'] for r in rows)
 timings.extend(r['elapsed'] for r in rows if 10<=r['step']<=50);ranks.append(dict(rank=rank,steps=len(by),final_step=max(by),scheduler_step=by[53]['scheduler_step']))
assert len(set(positions))==1
resume=Path('runs/ddp_real_subset_resume.log').read_text();assert resume.count('"resumed_step": 50')==8
pytest_log=Path('runs/pytest_final.log').read_text();assert ' passed' in pytest_log and 'failed' not in pytest_log and 'ERROR' not in pytest_log
closed=Path('runs/eval_mixed_subset_closed');manifest=json.loads((closed/'manifest.json').read_text());rows=[json.loads(x) for x in (closed/'predictions.jsonl').read_text().splitlines()]
assert len(rows)==576 and manifest['initial_pose_source']=='gt_first_frame' and manifest['full_sequences']
assert len({(r['stream_id'],r['frame_index']) for r in rows})==len(rows)
one=json.loads(Path('runs/eval_mixed_subset_onestep/metrics.json').read_text());assert all(m in one for m in ['learned','zero-motion','constant-velocity'])
c.update(engineering_preflight_passed=True,preflight_scope='one verified official s0 train sequence, 8 camera streams, 576 frames',
         official_s0_complete=False,preflight_approved=False,formal_training_blocker='Full extracted DexYCB and full s0 geometry/index preflight are unavailable',
         preflight_receipt='runs/subset_preflight_receipt.json')
c.pop('moving_center_per_sec',None);c.pop('moving_rotation_rad_per_sec',None);c['motion_speed_quantile']=.75
Path('configs/resolved_8gpu.yaml').write_text(yaml.safe_dump(c,sort_keys=False));Path('configs/resolved_8gpu.probe.json').write_text(json.dumps(probe,indent=2))
import numpy as np
receipt=dict(engineering_preflight_passed=True,official_s0_complete=False,formal_training_approved=False,
             data_scope=c['preflight_scope'],split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],ranks=ranks,
             global_samples_seen_reported_by_rank=positions,maximum_gpu_allocated_bytes=max(peaks),
             median_rank_optimizer_step_sec_after_warmup=float(np.median(timings)),
             pytest=pytest_log.strip(),overfit=overfit,closed_loop_rows=len(rows),
             unresolved=['Full s0 inventory/geometry/validation unavailable','Real FoundationPose inference not run','No held-out performance claim; this is a single-object training subset'],
             config_sha256=hashlib.sha256(Path('configs/resolved_8gpu.yaml').read_bytes()).hexdigest())
Path('runs/subset_preflight_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
