import pathlib,json,time,hashlib,os,subprocess
import numpy as np
r=pathlib.Path('/mnt/why/dexycb_lip');os.chdir(r);j=r/'runs/basin_boundary_v3';data=j/'data'
while not all((data/f'rank{i}.complete.json').exists() for i in range(4)):time.sleep(5)
parts=[]
for rank in range(4):
 with np.load(data/f'rank{rank}.npz') as z:parts.append({k:z[k].copy() for k in z.files})
merged={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};order=np.argsort(merged['frame_ids']);merged={k:v[order] for k,v in merged.items()};assert np.array_equal(merged['frame_ids'],np.arange(len(order)))
meta=[json.loads(str(x)) for x in merged['meta']];groups={s:{m['physical_sequence'] for m in meta if m['split']==s} for s in ['train','calibration','test']};assert not(groups['train']&groups['test'] or groups['calibration']&groups['test'] or groups['train']&groups['calibration'])
with np.load('runs/basin_v1/outcomes_v2/outcomes.npz') as old:
 used={json.loads(str(m))['physical_sequence'] for m in old['meta']};assert not used&groups['test']
 for i,m in enumerate(meta):
  if m['old_index'] is not None:
   assert np.array_equal(merged['y_converge'][i,:6],old['y_converge'][m['old_index']])
   assert np.array_equal(merged['candidate_pose'][i,:6],old['candidate_pose'][m['old_index']])
valid=merged['valid'];assert np.array_equal(merged['y_converge'][valid],merged['e_after'][valid]<.1);assert np.array_equal(merged['y_improve'][valid],merged['e_after'][valid]<merged['e_before'][valid]-.005)
p=data/'outcomes.npz';np.savez_compressed(p,**merged);h=hashlib.sha256(p.read_bytes()).hexdigest()
receipt=dict(passed=True,frames=len(order),candidates=int(valid.sum()),fresh_test_frames=sum(m['split']=='test' for m in meta),fresh_test_physical_sequences=len(groups['test']),data_sha256=h,source_first_six_preserved=True)
(data/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',PYTHONPATH=str(j/'candidate/src'),OMP_NUM_THREADS='2')
with (j/'fit.log').open('w') as log:rc=subprocess.call([str(r/'.venv-fp/bin/python'),'-u',str(j/'candidate/tools/train_basin_outcome.py'),'--spec',str(j/'candidate/configs/basin_boundary_v3.yaml'),'--out',str(j/'fit')],env=env,stdout=log,stderr=subprocess.STDOUT)
(j/'fit_exit.json').write_text(json.dumps(dict(returncode=rc)))
