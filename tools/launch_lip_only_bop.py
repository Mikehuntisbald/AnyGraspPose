"""Launch the validated independent LIP experiment; no FP inference process."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha
from lip.benchmark.bop_io import load_predictions
from streaming_bop_utils import plan_streams

root=Path(__file__).resolve().parents[1]
run=root/'runs/lip_only_s0_test'
assert not (run/'launch.json').exists()
p=json.loads((run/'protocol.json').read_text())
smoke=json.loads((root/'runs/lip_only_smoke/manifest.json').read_text())
if p.get('fixture_only',False) or smoke.get('fixture_only',False):raise ValueError('Interface fixtures cannot approve an official benchmark')
if smoke.get('device')!='cuda':raise ValueError('Official launch requires its GPU smoke')
if smoke['protocol_sha256']!=sha(run/'protocol.json') or smoke['source_sha256']!=p['source_sha256']:raise ValueError('Official smoke provenance mismatch')
access=smoke['access_audit']['counts']
if any(value for key,value in access.items() if key.startswith('denied_')):raise ValueError('Official smoke attempted forbidden reads')
if access['validated_native_image_reads']!=2*smoke['processed_object_frames']:raise ValueError('Official smoke image-read audit is incomplete')
assert smoke['completed'] and not smoke['failures'] and smoke['fp_calls']==0 and not smoke['fp_modules_imported']
assert smoke['max_history_frames']==8 and smoke['max_ablated_history_frames']==0
plans=plan_streams(json.loads(Path(p['targets']).read_text()),load_predictions(p['initializer_csv']))
inits={(x['scene_id'],x['obj_id']):x['initializer'] for x in plans}
compared=0
for method in p['methods']:
    for row in load_predictions(root/'runs/lip_only_smoke'/(method+'.csv')):
        init=inits[row['scene_id'],row['obj_id']]
        if row['im_id']==init['im_id']:
            np.testing.assert_allclose(row['pose'],init['pose'].astype('f4'),rtol=0,atol=1e-12)
            compared+=1
assert compared==4
(run/'standalone_preflight.json').write_text(json.dumps(dict(completed=True,real_smoke=smoke,
    initial_export_matches_direct_posecnn=compared,entrypoint_sha256=sha(root/'tools/infer_lip_tracking_bop.py')),indent=2))
launch=[]
for rank in range(32):
    cmd=[sys.executable,str(root/'tools/infer_lip_tracking_bop.py'),'--protocol',str(run/'protocol.json'),
         '--out',str(run/f'rank{rank}'),'--rank',str(rank),'--world','32']
    with (run/f'rank{rank}.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank%8),
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(root/'src')),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch.append(dict(rank=rank,pid=proc.pid,gpu=rank%8,command=cmd))
(run/'launch.json').write_text(json.dumps(launch,indent=2))
(run/'dependency_receipt.json').write_text(json.dumps(dict(fp_used=False,
    helpers_sha256={f.name:sha(f) for f in (root/'tools').glob('*.py')}),indent=2))
cmd=[sys.executable,str(root/'tools/finish_streaming_bop.py'),'--run',str(run),
     '--toolkit','/mnt/why/dexycb_lip/third_party/dex-ycb-toolkit-official',
     '--bop-python','/mnt/why/dexycb_lip/.venv-bop/bin/python']
with (run/'supervisor.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,PYTHONPATH=str(root/'src')),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/'supervisor.json').write_text(json.dumps(dict(pid=proc.pid,command=cmd),indent=2))
print(json.dumps(dict(workers=32,supervisor_pid=proc.pid,fp_calls=0,run=str(run))))
