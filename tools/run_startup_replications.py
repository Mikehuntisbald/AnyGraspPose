"""Two independent startup factorials, sequentially using all eight GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lip.engine.stream_checkpoint import source_hash


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--experiment',type=Path,action='append',required=True)
    parser.add_argument('--hold-audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();root=Path(__file__).resolve().parents[1]
    assert len(a.experiment)==2
    original=json.loads((a.original/'experiment.json').read_text())
    experiments=[json.loads((p/'experiment.json').read_text()) for p in a.experiment]
    assert [e['seed'] for e in experiments]==[1000003,2000003]
    assert all(e['source_sha256']==source_hash()==original['source_sha256'] for e in experiments)
    a.out.mkdir(parents=True,exist_ok=False)
    scripts=['run_startup_replications.py','verify_startup_replication_inputs.py','compare_startup_replications.py',
             'prepare_startup_factorial.py','run_startup_factorial.py','evaluate_startup_factorial.py',
             'verify_startup_start.py','compare_startup_factorial.py','analyze_pose_robustness.py',
             'analyze_reference_feedback.py','audit_bad_initial_errors.py','compare_rk_ablation.py']
    paths=[root/'tools'/name for name in scripts]
    paths.append(root/'tests/test_startup_replication_statistics.py')
    for folder,e in zip(a.experiment,experiments):
        paths.extend([folder/'experiment.json',folder/'sampling_receipt.json',Path(e['training_manifest'])])
        for arm in e['arms'].values():paths.extend([Path(arm['config']),Path(arm['init'])])
    bound={str(p):digest(p) for p in paths}
    status=dict(pid=os.getpid(),phase='preparing',commands=[],source_sha256=source_hash(),bindings=bound,
                scope='Two additional independent adaptation seeds. Each runs all four predeclared startup arms for 1000 steps with full matched s0 val, eight GPUs, same frozen ancestor and code. No architecture additions, official test access or checkpoint promotion.')
    env=dict(os.environ,PYTHONPATH=str(root/'tools')+os.pathsep+str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    child=None
    def save(phase,**fields):
        status.update(phase=phase,updated=time.time(),**fields)
        temporary=a.out/'status.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(a.out/'status.json')
    def verify_bound():
        assert source_hash()==status['source_sha256']
        assert all(digest(Path(path))==sha for path,sha in bound.items())
    def run(name,command,gpus=''):
        nonlocal child
        verify_bound()
        with (a.out/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=root,env=dict(env,CUDA_VISIBLE_DEVICES=gpus),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status['commands'].append(dict(name=name,command=command,pid=child.pid,log=str(a.out/(name+'.log'))))
            save(name)
            result=child.wait();child=None
        if result:raise RuntimeError(name+' exited '+str(result))
    try:
        save('preparing')
        run('statistics_tests',[sys.executable,'-m','pytest','tests/test_startup_replication_statistics.py','-q','--junitxml='+str(a.out/'statistics_tests.xml')])
        run('input_audit',[sys.executable,'tools/verify_startup_replication_inputs.py','--original',str(a.original),
            *[x for folder in a.experiment for x in ('--experiment',str(folder))],'--out',str(a.out/'input_audit.json')])
        for folder,e in zip(a.experiment,experiments):
            name='seed_'+str(e['seed'])
            run(name,[sys.executable,'tools/run_startup_factorial.py','--out',str(folder)],'0,1,2,3,4,5,6,7')
            assert json.loads((folder/'status.json').read_text())['phase']=='completed'
            evaluations={'parent':e['reference_evaluation'],**{arm:str(folder/arm/'s0_val') for arm in e['arms']}}
            flags=[x for arm,path in evaluations.items() for x in ('--evaluation',arm+'='+path)]
            run(name+'_robustness',[sys.executable,'tools/analyze_pose_robustness.py',*flags,
                '--reference','S0O0','--out',str(a.out/(name+'_robustness'))])
            run(name+'_fixed_groups',[sys.executable,'tools/analyze_reference_feedback.py',*flags,
                '--hold-audit',str(a.hold_audit),'--out',str(a.out/(name+'_fixed_groups'))])
            run(name+'_bad_initial',[sys.executable,'tools/audit_bad_initial_errors.py',
                '--reference-eval',str(folder/'S0O0/s0_val'),'--candidate-eval',str(folder/'S1O1/s0_val'),
                '--index-root',e['index_root'],'--out',str(a.out/(name+'_bad_initial'))])
        run('combined',[sys.executable,'tools/compare_startup_replications.py',
            *[x for folder in (a.original,*a.experiment) for x in ('--experiment',str(folder))],
            '--out',str(a.out/'combined')])
        verify_bound();save('completed')
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save('failed',error=repr(error));raise


if __name__=='__main__':main()
