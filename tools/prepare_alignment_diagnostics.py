"""Freeze a train-only diagnostic cohort before probes or short fitting."""
import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from lip.data.stream_clips import StreamClips
from lip.data.external_initializers import request_real_initializer,initializer_key
from lip.geometry.so3 import angle


def main():
 root=Path(__file__).resolve().parents[1];out=root/'runs/train_diagnostic';out.mkdir(exist_ok=False);torch.set_num_threads(2)
 original=Path('/mnt/why/dexycb_lip/intraframe_alignment_20260915/runs/alignment_pair');e=json.loads((original/'experiment.json').read_text());assert source_hash()==e['source_sha256']
 fixed=json.loads(Path(e['training_manifest']).read_text());assert sha(e['training_manifest'])==e['training_manifest_sha256']
 data=StreamClips(e['data_root'],e['index_root'],8,48,fixed=fixed,decode_threads=0,
  external_initializers=e['train_initializers'],external_initializers_sha256=e['train_initializers_sha256'],real_initialization_probability=.5,include_initial_observation=True)
 bins=((0,15),(15,45),(45,90),(90,181));chosen={i:[] for i in range(4)};seen=set();objects={i:set() for i in range(4)}
 for i,item in enumerate(fixed):
  if not request_real_initializer(item['seed'],.5):continue
  stream=data.streams[item['stream']];physical=stream['subject_id']+'/'+stream['sequence_id']
  if physical in seen:continue
  frame=int(data.poses[item['stream']]['frames'][item['start']]);entry=data.external[initializer_key(stream['stream_id'],frame)]
  if entry is None:continue
  gt=data.poses[item['stream']]['poses'][item['start']][:3,:3];r=np.asarray(entry['pose_original'],dtype='f4')[:3,:3];degrees=float(angle(torch.from_numpy(r@gt.T))*180/torch.pi)
  b=next(n for n,(lo,hi) in enumerate(bins) if lo<=degrees<hi)
  if len(chosen[b])>=4 or stream['object_id'] in objects[b]:continue
  chosen[b].append(dict(manifest_index=i,sample=item,stream_id=stream['stream_id'],physical_sequence=physical,object_id=stream['object_id'],initial_rotation_deg=degrees,
    bin=list(bins[b]),pose_cache_sha256=sha(Path(e['index_root'])/stream['pose_cache'])))
  seen.add(physical);objects[b].add(stream['object_id'])
  if all(len(v)==4 for v in chosen.values()):break
 assert all(len(v)==4 for v in chosen.values())
 cohort=dict(fit=[v for b in chosen.values() for v in b[:2]],probe=[v for b in chosen.values() for v in b[2:]])
 result=dict(completed=True,split='train',selection='First four unique physical sequences and object IDs per initial-angle bin among real-initializer requests in the frozen train manifest; first two fit, next two probe. No val/test selection or supervision.',
  source_sha256=source_hash(),original_experiment=str(original/'experiment.json'),original_experiment_sha256=sha(original/'experiment.json'),
  configuration=e['arms']['alignment']['config'],initial_checkpoint=e['arms']['alignment']['init'],initial_checkpoint_sha256=e['arms']['alignment']['init_sha256'],
  final_checkpoint=str(original/'alignment/train/last.pt'),final_checkpoint_sha256=sha(original/'alignment/train/last.pt'),
  data_root=e['data_root'],index_root=e['index_root'],training_manifest=e['training_manifest'],training_manifest_sha256=e['training_manifest_sha256'],
  train_initializers=e['train_initializers'],train_initializers_sha256=e['train_initializers_sha256'],cohort=cohort,
  short_fit=dict(steps=64,arms=['frozen_parent','joint'],batch_clips=8,precision='bf16',branch_lr=1e-4,loaded_lr=2e-6,other_new_lr=1e-5,warmup=1,evaluate_steps=[0,16,64],
   purpose='Bounded train-only learnability diagnostic from identical zero-branch initialization. Fresh optimizer, repeated eight fit clips; eight distinct train probe clips never optimized. Not a candidate training run or val gain claim.'))
 (out/'spec.json').write_text(json.dumps(result,indent=2));print(json.dumps(cohort,indent=2))

if __name__=='__main__':main()
