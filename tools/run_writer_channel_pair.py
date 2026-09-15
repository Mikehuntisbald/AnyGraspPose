"""Evaluate frozen rotation-only and center-only writer masks after seed replication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('runtime','wait-root','checkpoint','config','initial-poses','data-root','index-root','on-eval','off-eval','out'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--wait-pid',required=True,type=int);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.runtime/'src'))
    from lip.engine.stream_checkpoint import source_hash
    from lip.engine.stream_config import load_stream_config
    m=json.loads((a.on_eval/'manifest.json').read_text())
    assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
    assert m['architecture_id']=='stream_rk_adaptive_reference' and m['checkpoint_sha256']==digest(a.checkpoint)
    assert m['source_sha256']==source_hash() and m['initial_poses_sha256']==digest(a.initial_poses)
    assert load_stream_config(a.config)==m['config']
    for folder,mode in ((a.on_eval,'learned'),(a.off_eval,'zero')):
        other=json.loads((folder/'manifest.json').read_text())
        assert other['completed'] and other['population_verified'] and other['frames']==23200 and len(other['streams'])==320
        assert all(other[key]==m[key] for key in ('checkpoint_sha256','source_sha256','config','split_hash','mesh_hash','initial_poses_sha256'))
        intervention=other['inference_intervention']
        assert intervention['mode']==mode and intervention['component']=='writer' and intervention.get('zero_channels','both')=='both'
        assert intervention['weights_bitwise_unchanged'] and intervention['source_and_intervention_bound']
        assert other['fp_calls']==other['critic_calls']==0
    paths=[Path(__file__),a.checkpoint,a.config,a.initial_poses,
        *[root/'tools'/name for name in ('evaluate_reference_bypass.py','analyze_pose_robustness.py','audit_bad_initial_errors.py','compare_rk_ablation.py')],
        *[folder/name for folder in (a.on_eval,a.off_eval) for name in ('manifest.json','predictions.jsonl')]]
    bound={str(path):digest(path) for path in paths}
    status=dict(pid=os.getpid(),phase='preparing',commands=[],bindings=bound,source_sha256=source_hash(),
        waiting_for_pid=a.wait_pid,waiting_for_root=str(a.wait_root),
        scope='Frozen final adaptive checkpoint only. After both pending seeded factorials complete, run R0C1 (rotation write zero) and R1C0 (center write zero), 320 streams / 23200 frames each on four GPUs. Reuse finalized R1C1/R0C0 trajectories. No retraining or candidate promotion; same controlled noisy-GT initializer.')
    jobs=[]
    def save(phase,**extra):
        status.update(phase=phase,updated=time.time(),**extra)
        temporary=a.out/'status.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(a.out/'status.json')
    def verify_bound():
        assert source_hash()==status['source_sha256']
        assert all(digest(Path(path))==expected for path,expected in bound.items())
    env=dict(os.environ,PYTHONPATH=str(root/'tools')+':'+str(a.runtime/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def run(name,command):
        status['commands'].append(dict(name=name,command=command));save(name)
        with (a.out/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=a.runtime,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        starttime=None
        while True:
            upstream=json.loads((a.wait_root/'status.json').read_text())
            if upstream['phase']=='completed':
                combined=json.loads((a.wait_root/'combined/comparison.json').read_text())
                assert combined['completed'] and combined['seeds']==[42,1000003,2000003]
                break
            if upstream['phase']=='failed':raise RuntimeError('Seed replication failed')
            process=Path('/proc')/str(a.wait_pid)
            command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            current=(process/'stat').read_text().rsplit(')',1)[1].split()[19]
            assert 'run_startup_replications.py' in command and str(a.wait_root) in command
            assert starttime is None or current==starttime
            starttime=current;save('waiting',upstream_phase=upstream['phase'],upstream_starttime=starttime)
            time.sleep(15)
        verify_bound()
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError('GPUs are occupied: '+active)
        assert len(subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines())==8
        for index,(name,channel) in enumerate((('R0C1','rotation'),('R1C0','center'))):
            folder=a.out/name;folder.mkdir()
            for rank in range(4):
                command=[sys.executable,str(root/'tools/evaluate_reference_bypass.py'),'--runtime',str(a.runtime),
                    '--mode','zero','--component','writer','--zero-channels',channel,
                    '--config',str(a.config),'--checkpoint',str(a.checkpoint),'--initial-poses',str(a.initial_poses),
                    '--data-root',str(a.data_root),'--index-root',str(a.index_root),'--split','val',
                    '--rank',str(rank),'--world-size','4','--out',str(folder/f'rank{rank}')]
                log=(folder/f'rank{rank}.log').open('w')
                proc=subprocess.Popen(command,cwd=a.runtime,env=dict(env,CUDA_VISIBLE_DEVICES=str(index*4+rank)),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                jobs.append((proc,log));status['commands'].append(dict(name=name,channel=channel,rank=rank,pid=proc.pid,command=command))
        while any(proc.poll() is None for proc,log in jobs):
            if any(proc.poll() not in (None,0) for proc,log in jobs):raise RuntimeError('Channel evaluation failed')
            save('evaluating');time.sleep(5)
        for proc,log in jobs:assert proc.returncode==0;log.close()
        for name,channel in (('R0C1','rotation'),('R1C0','center')):
            run(name+'_merge',[sys.executable,str(root/'tools/evaluate_reference_bypass.py'),'--runtime',str(a.runtime),
                '--mode','zero','--component','writer','--zero-channels',channel,'--merge','--world-size','4','--out',str(a.out/name)])
        folders={'R1C1':a.on_eval,'R0C0':a.off_eval,'R0C1':a.out/'R0C1','R1C0':a.out/'R1C0'}
        flags=[arg for name,folder in folders.items() for arg in ('--evaluation',name+'='+str(folder))]
        run('analysis',[sys.executable,str(root/'tools/analyze_pose_robustness.py'),*flags,'--reference','R1C1','--out',str(a.out/'analysis')])
        for name in ('R0C1','R1C0'):
            run(name+'_bad_initial',[sys.executable,str(root/'tools/audit_bad_initial_errors.py'),
                '--reference-eval',str(a.on_eval),'--candidate-eval',str(a.out/name),
                '--index-root',str(a.index_root),'--out',str(a.out/(name+'_bad_initial'))])
        verify_bound();save('completed')
    except BaseException as error:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,log in jobs:proc.wait();log.close()
        save('failed',error=repr(error));raise


if __name__=='__main__':main()
