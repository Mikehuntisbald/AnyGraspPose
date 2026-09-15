"""Launch balanced, GPU-smoke-verified standalone inference and official scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from lip.benchmark.bop_io import load_predictions
from streaming_bop_utils import plan_streams
from accelerate_streaming_bop import assign_balanced,identity
from standalone_bop_state import standalone_methods


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('run','smoke','toolkit'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bop-python',required=True);p.add_argument('--gpus',required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];run=a.run.resolve();protocol=run/'protocol.json'
    spec=json.loads(protocol.read_text());methods=standalone_methods(spec)
    assert not (run/'launch.json').exists() and spec['frozen'] and spec['fp_calls']==0
    assert source_hash()==spec['source_sha256'] and sha(spec['freeze_receipt'])==spec['freeze_sha256']
    assert sha(spec['checkpoint'])==spec['checkpoint_sha256'] and sha(spec['config'])==spec['config_sha256']
    assert all(sha(root/'tools'/name)==value for name,value in spec['tools_sha256'].items())
    smoke=json.loads((a.smoke/'manifest.json').read_text())
    assert not spec.get('fixture_only',False) and not smoke['fixture_only'] and smoke['device']=='cuda'
    assert smoke['protocol_sha256']==sha(protocol) and smoke['source_sha256']==spec['source_sha256']
    assert smoke['completed'] and not smoke['failures'] and smoke['fp_calls']==0 and not smoke['fp_modules_imported']
    assert smoke['methods']==list(methods) and smoke['max_ablated_history_frames']==0
    assert 'lip_temporal' not in methods or smoke['max_history_frames']==8
    access=smoke['access_audit']['counts']
    assert not any(v for k,v in access.items() if k.startswith('denied_'))
    assert access['validated_native_image_reads']==2*smoke['processed_object_frames']
    plans=plan_streams(json.loads(Path(spec['targets']).read_text()),load_predictions(spec['initializer_csv']))
    inits={(x['scene_id'],x['obj_id']):x['initializer'] for x in plans};compared=0
    for method in methods:
        assert sha(a.smoke/(method+'.csv'))==smoke['csv_sha256'][method]
        for row in load_predictions(a.smoke/(method+'.csv')):
            init=inits[row['scene_id'],row['obj_id']]
            if row['im_id']==init['im_id']:
                np.testing.assert_allclose(row['pose'],init['pose'].astype('f4'),rtol=0,atol=1e-12);compared+=1
    assert compared==2*len(methods)
    gpus=[int(x) for x in a.gpus.split(',')];assert len(gpus)==len(set(gpus)) and a.workers>=len(gpus)>0
    for gpu in gpus:
        active=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError('Assigned GPU already has a compute process: '+str(gpu))
    assignment,loads=assign_balanced(plans,a.workers);flat=[tuple(x) for values in assignment.values() for x in values]
    assert len(flat)==len(set(flat)) and set(flat)=={identity(x) for x in plans}
    assert sum(loads)==spec['population']['object_frames']
    assignments=run/'assignments.json';assignments.write_text(json.dumps(assignment,indent=2))
    execution=run/'execution.json';execution.write_text(json.dumps(dict(protocol_sha256=sha(protocol),workers=a.workers,
        gpus=gpus,assignments=str(assignments),assignments_sha256=sha(assignments),worker_frame_loads=loads,
        tools_sha256=spec['tools_sha256'],inference_math_changed=False),indent=2))
    (run/'standalone_preflight.json').write_text(json.dumps(dict(completed=True,real_smoke=smoke,
        initial_export_matches_direct_posecnn=compared,launch_sha256=sha(Path(__file__))),indent=2))
    launch=[];processes=[]
    try:
        for rank in range(a.workers):
            gpu=gpus[rank%len(gpus)]
            cmd=[sys.executable,str(root/'tools/infer_lip_tracking_bop.py'),'--protocol',str(protocol),'--execution',str(execution),
                '--out',str(run/f'rank{rank}'),'--rank',str(rank),'--world',str(a.workers)]
            with (run/f'rank{rank}.log').open('x') as log:
                proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(root/'src')),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(proc);launch.append(dict(rank=rank,pid=proc.pid,gpu=gpu,command=cmd))
        (run/'launch.json').write_text(json.dumps(launch,indent=2))
        cmd=[sys.executable,str(root/'tools/finish_streaming_bop.py'),'--run',str(run),'--toolkit',str(a.toolkit),'--bop-python',a.bop_python]
        with (run/'supervisor.log').open('x') as log:
            supervisor=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,PYTHONPATH=str(root/'src')),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (run/'supervisor.json').write_text(json.dumps(dict(pid=supervisor.pid,command=cmd),indent=2))
    except BaseException:
        for proc in processes:
            if proc.poll() is None:proc.terminate()
        for proc in processes:proc.wait()
        raise
    print(json.dumps(dict(workers=a.workers,supervisor_pid=supervisor.pid,fp_calls=0,run=str(run))))


if __name__=='__main__':main()
