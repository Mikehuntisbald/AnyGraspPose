import pathlib,json,time,hashlib,subprocess,os
r=pathlib.Path('/mnt/why/dexycb_lip');os.chdir(r);j=r/'runs/basin_v1';data=j/'data'
while not all((data/f'complete_rank{i}.json').exists() for i in range(4)):time.sleep(5)
files=sorted(data.glob('frame_[0-9][0-9][0-9][0-9][0-9].npz'));assert len(files)==2048
import numpy as np
h=hashlib.sha256();keys=set();counts={};positive=0
for f in files:
 h.update(f.name.encode());h.update(f.read_bytes())
 with np.load(f) as z:
  m=json.loads(str(z['meta']));k=(m['stream_id'],m['frame_index']);assert k not in keys;keys.add(k);positive+=int(z['labels'].sum());counts[m['split']]=counts.get(m['split'],0)+1
(data/'complete.json').write_text(json.dumps(dict(frames=len(files),candidates=len(files)*6,positives=positive,split_frames=counts,data_sha256=h.hexdigest()),indent=2))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',PYTHONPATH=str(j/'candidate/src'),OMP_NUM_THREADS='2')
with (j/'fit.log').open('w') as log:
 rc=subprocess.call([str(r/'.venv-fp/bin/python'),'-u',str(j/'candidate/tools/train_basin_critic.py'),'--data',str(data),'--out',str(j/'fit')],env=env,stdout=log,stderr=subprocess.STDOUT)
(j/'fit_exit.json').write_text(json.dumps(dict(returncode=rc)))
