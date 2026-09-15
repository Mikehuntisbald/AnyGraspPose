"""Queue a frozen learned/zero feedback pair after an existing factorial finishes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upstream_status(experiment,controller_pid,expected_starttime=None,controller_script='run_startup_factorial.py'):
    if controller_script not in ('run_startup_factorial.py','run_spatial_memory.py'):raise ValueError('Unsupported upstream controller')
    state=json.loads((experiment/'status.json').read_text())
    if state['phase']=='completed':
        if not json.loads((experiment/'comparison.json').read_text())['completed']:
            raise RuntimeError('Factorial comparison incomplete')
        return 'completed',expected_starttime
    if state['phase']=='failed':raise RuntimeError('Upstream factorial failed')
    handle=Path('/proc')/str(controller_pid)
    command=(handle/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    current=(handle/'stat').read_text().rsplit(')',1)[1].split()[19]
    if controller_script not in command or str(experiment) not in command or (expected_starttime is not None and current!=expected_starttime):
        raise RuntimeError('Upstream process identity changed')
    return state['phase'],current


def check_repeat(archived, repeated):
    """Require the learned-mode rerun to reproduce the saved causal trajectory."""
    data=[];manifests=[]
    for folder in (archived,repeated):
        m=json.loads((folder/'manifest.json').read_text())
        if not (m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320):
            raise ValueError('Full 320-stream evaluation required')
        if m['split']!='val' or m['fp_calls'] or m['critic_calls']:raise ValueError('Wrong protocol')
        rows=list(map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()))
        keyed={(r['stream_id'],r['frame_index']):r for r in rows}
        if len(keyed)!=len(rows) or len(rows)!=23200:raise ValueError('Duplicate or missing frames')
        data.append(keyed);manifests.append(m)
    for key in ('checkpoint_sha256','source_sha256','split_hash','mesh_hash','initial_poses_sha256','config'):
        if manifests[0][key]!=manifests[1][key]:raise ValueError('Repeat provenance differs: '+key)
    if set(data[0])!=set(data[1]):raise ValueError('Repeat target keys differ')
    maximum=0.;changed=0
    for key,row in data[0].items():
        other=data[1][key]
        for field in ('initialization','object_id','visibility','status','add_01','adds_005'):
            if row[field]!=other[field]:raise ValueError('Repeat result differs: '+str((key,field)))
        delta=max(abs(x-y) for a,b in zip(row['pose_centered'],other['pose_centered']) for x,y in zip(a,b))
        maximum=max(maximum,delta);changed+=int(delta!=0)
    if maximum>1e-6:raise ValueError('Learned repeat pose tolerance exceeded: '+str(maximum))
    return dict(completed=True,frames=23200,streams=320,max_pose_element_abs_difference=maximum,
        changed_pose_rows=changed,tolerance=1e-6,discrete_outcomes_identical=True,
        archived_prediction_sha256=digest(archived/'predictions.jsonl'),repeated_prediction_sha256=digest(repeated/'predictions.jsonl'))


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('runtime','checkpoint','archived-eval','config','initial-poses','data-root','index-root','wait-experiment','out'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--controller-pid',required=True,type=int)
    p.add_argument('--controller-script',default='run_startup_factorial.py',choices=('run_startup_factorial.py','run_spatial_memory.py'))
    p.add_argument('--component',default='feedback',choices=('feedback','writer'));a=p.parse_args()
    root=Path(__file__).resolve().parents[1];wrapper=root/'tools/evaluate_reference_bypass.py'
    sys.path.insert(0,str(a.runtime/'src'))
    from lip.engine.stream_checkpoint import source_hash
    a.out.mkdir(parents=True,exist_ok=False)
    status=dict(pid=os.getpid(),phase='preparing',arguments={k:str(v) for k,v in vars(a).items()},commands=[],
        scope='Frozen same-weight output intervention, full controlled val, shared noisy-GT initializer, zero FP. No training. This does not evaluate official non-GT BOP AR.')
    jobs=[]
    def save(phase,**extra):
        status.update(phase=phase,updated=time.time(),**extra);temp=a.out/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(a.out/'status.json')
    def run(command,log_path,env):
        with log_path.open('w') as log:subprocess.run(command,cwd=a.runtime,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        m=json.loads((a.archived_eval/'manifest.json').read_text())
        bindings={str(path):digest(path) for path in (a.checkpoint,a.config,a.initial_poses,wrapper,Path(__file__),
            root/'tools/analyze_pose_robustness.py',root/'tools/audit_bad_initial_errors.py',root/'tools/compare_rk_ablation.py',
            a.archived_eval/'manifest.json',a.archived_eval/'predictions.jsonl')}
        if bindings[str(a.checkpoint)]!=m['checkpoint_sha256'] or source_hash()!=m['source_sha256']:
            raise ValueError('Checkpoint/runtime differs from archived candidate')
        if bindings[str(a.initial_poses)]!=m['initial_poses_sha256']:raise ValueError('Initializer differs')
        status.update(bindings=bindings,base_source_sha256=source_hash())
        starttime=None
        while True:
            phase,starttime=upstream_status(a.wait_experiment,a.controller_pid,starttime,a.controller_script)
            if phase=='completed':break
            save('waiting',upstream_phase=phase);time.sleep(15)
        if source_hash()!=status['base_source_sha256'] or any(digest(Path(path))!=sha for path,sha in bindings.items()):
            raise RuntimeError('Bound input changed while waiting')
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError('GPUs still occupied: '+active)
        gpu_count=len(subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines())
        if gpu_count!=8:raise RuntimeError('Expected eight GPUs')
        env=dict(os.environ,PYTHONPATH=str(a.runtime/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        for mode_index,mode in enumerate(('learned','zero')):
            folder=a.out/mode;folder.mkdir()
            for rank in range(4):
                gpu=4*mode_index+rank
                command=[sys.executable,str(wrapper),'--runtime',str(a.runtime),'--mode',mode,'--component',a.component,'--config',str(a.config),
                    '--checkpoint',str(a.checkpoint),'--out',str(folder/f'rank{rank}'),'--data-root',str(a.data_root),
                    '--index-root',str(a.index_root),'--split','val','--rank',str(rank),'--world-size','4','--initial-poses',str(a.initial_poses)]
                log=(folder/f'rank{rank}.log').open('w')
                proc=subprocess.Popen(command,cwd=a.runtime,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                jobs.append((proc,log));status['commands'].append(dict(mode=mode,rank=rank,gpu=gpu,pid=proc.pid,command=command))
        while any(proc.poll() is None for proc,log in jobs):
            if any(proc.poll() not in (None,0) for proc,log in jobs):raise RuntimeError('Intervention shard failed')
            save('evaluating');time.sleep(5)
        for proc,log in jobs:
            if proc.returncode:raise RuntimeError('Intervention shard failed')
            log.close()
        cpu=dict(env,CUDA_VISIBLE_DEVICES='')
        for mode in ('learned','zero'):
            run([sys.executable,str(wrapper),'--runtime',str(a.runtime),'--mode',mode,'--component',a.component,'--merge','--world-size','4','--out',str(a.out/mode)],a.out/(mode+'_merge.log'),cpu)
        save('checking_learned_repeat')
        repeat=check_repeat(a.archived_eval,a.out/'learned');(a.out/'repeat_receipt.json').write_text(json.dumps(repeat,indent=2))
        save('analyzing')
        run([sys.executable,str(root/'tools/analyze_pose_robustness.py'),'--evaluation','learned='+str(a.out/'learned'),
            '--evaluation','zero='+str(a.out/'zero'),'--reference','learned','--out',str(a.out/'analysis')],a.out/'analysis.log',cpu)
        run([sys.executable,str(root/'tools/audit_bad_initial_errors.py'),'--reference-eval',str(a.out/'learned'),
            '--candidate-eval',str(a.out/'zero'),'--index-root',str(a.index_root),'--out',str(a.out/'bad_initial_errors')],a.out/'bad_initial_errors.log',cpu)
        save('completed')
    except BaseException as error:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,log in jobs:
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        save('failed',error=repr(error));raise


if __name__=='__main__':main()
