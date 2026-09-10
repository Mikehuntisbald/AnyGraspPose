"""Seal the actual subject-subset preflight; do not approve full-s0 training."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,yaml
p=argparse.ArgumentParser();p.add_argument('--out',default='runs/available_20260910');p.add_argument('--index',default='cache/s0_available_20260910');p.add_argument('--candidate',default='configs/candidate_available_20260910.yaml');a=p.parse_args()
out=Path(a.out);index=Path(a.index);read=lambda p:json.loads(Path(p).read_text())
audit=read(index/'audit.json');geo=read(index/'geometry_gate.json');overfit=read(out/'overfit32/report.json');probe=read(Path(a.candidate).with_suffix('.probe.json'))
assert audit['verified_subject_subset'] and not audit['complete'] and not audit['errors'] and audit['split_disjoint']
assert geo['passed'] and set(s['object_id'] for s in geo['samples'])==set(audit['object_ids'])
assert overfit['passed'] and overfit['steps']==500 and overfit['num_clips']==32
assert probe['source']=='real_DexYCB' and not probe['data_complete']
ranks=[];times=[];peaks=[]
for rank in range(8):
 rows=[json.loads(x) for x in (out/f'ddp/rank{rank}.jsonl').read_text().splitlines()];by={r['step']:r for r in rows}
 assert all(s in by for s in range(1,54))
 assert all(r['step']==r['scheduler_step']==r['sampler_position'] and r['rollout']==4 and r['effective_batch']==256 and r['nonfinite_count']==0 for r in rows)
 ranks.append(dict(rank=rank,global_step=by[53]['step'],scheduler_step=by[53]['scheduler_step'],samples_seen=by[53]['samples_seen']))
 times.extend(r['elapsed'] for r in rows if 10<=r['step']<=50);peaks.extend(r['peak_memory_bytes'] for r in rows)
assert (out/'ddp_resume.log').read_text().count('"resumed_step": 50')==8
pytest=(out/'pytest.log').read_text();assert ' passed' in pytest and 'failed' not in pytest and 'ERROR' not in pytest
manifest=read(out/'val_closed/manifest.json');pred=[json.loads(x) for x in (out/'val_closed/predictions.jsonl').read_text().splitlines()]
streams=[json.loads(x) for x in (index/'streams.jsonl').read_text().splitlines()]
lookup={s['stream_id']:s for s in streams};selected=manifest['streams']
assert manifest['split']=='val' and manifest['initial_pose_source']=='gt_first_frame' and manifest['full_sequences']
assert manifest['checkpoint']['sha256'] and len(selected)==20 and all(lookup[s]['split']=='val' for s in selected)
assert len(pred)==sum(lookup[s]['num_frames'] for s in selected)
assert len({(r['stream_id'],r['frame_index']) for r in pred})==len(pred)
assert set(r['object_id'] for r in pred)==set(audit['object_ids'])
one=read(out/'val_onestep/metrics.json');assert all(one[m]['micro']['count']==32 for m in ['learned','zero-motion','constant-velocity'])
c=yaml.safe_load(Path(a.candidate).read_text());assert c['batch_size_per_gpu'] in probe['eligible_batches'] and c['batch_size_per_gpu']*c['grad_accum_steps']*8==256
c.update(engineering_preflight_passed=True,preflight_approved=False,official_s0_complete=False,
         preflight_scope='Complete subjects 02,03,10; official s0 train/val/test membership retained',
         selected_subjects=audit['selected_subjects'],data_root_hint=audit['source_root'],index_root_hint=str(index),
         preflight_receipt=str(out/'receipt.json'),formal_training_blocker='Remaining subjects are not included; full-s0 preflight is pending')
resolved=Path('configs/resolved_8gpu.yaml')
if resolved.exists():
 old=resolved.read_bytes();backup=out/'previous_resolved_8gpu.yaml'
 if not backup.exists():backup.write_bytes(old)
resolved.write_text(yaml.safe_dump(c,sort_keys=False));resolved.with_suffix('.probe.json').write_text(json.dumps(probe,indent=2))
r=dict(engineering_preflight_passed=True,formal_training_approved=False,scope=c['preflight_scope'],index=audit,
       geometry_samples=len(geo['samples']),geometry_objects=len(set(s['object_id'] for s in geo['samples'])),
       depth_median_mm=float(np.median([s['depth_abs_median_m'] for s in geo['samples']])*1000),
       crop_max_px=max(s['crop_projection_max'] for s in geo['samples']),overfit=overfit,ranks=ranks,
       peak_gpu_allocated_bytes=max(peaks),median_rank_step_seconds=float(np.median(times)),pytest=pytest.strip(),
       validation_streams=len(selected),validation_frames=len(pred),validation_checkpoint=manifest['checkpoint'],
       config_sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),no_long_training_launched=True)
(out/'receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
