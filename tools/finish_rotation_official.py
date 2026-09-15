"""After controlled validation: freeze once, evaluate two complete standalone models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('policy','diagnostics','interface-receipt','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--diagnostics-pid',required=True,type=int);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    from lip.engine.stream_checkpoint import source_hash
    policy=json.loads(a.policy.read_text());interface=json.loads(a.interface_receipt.read_text())
    assert interface['completed'] and interface['source_sha256']==source_hash()==policy['source_sha256']
    bound={str(path):sha(path) for path in [a.policy,a.interface_receipt,*sorted((root/'tools').glob('*.py'))]}
    assert all(sha(root/'tools'/name)==value for name,value in interface['tools_sha256'].items())
    a.out.mkdir(parents=True,exist_ok=False);status=dict(pid=os.getpid(),phase='waiting',bindings=bound,source_sha256=source_hash())
    env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');current=None
    def save(phase,**extra):
        status.update(phase=phase,updated=time.time(),**extra);tmp=a.out/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(a.out/'status.json')
    def run(name,cmd,gpu=''):
        nonlocal current
        save(name,command=cmd)
        with (a.out/(name+'.log')).open('x') as log:
            current=subprocess.Popen(cmd,cwd=root,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=current.pid;save(name);code=current.wait();current=None
            if code:raise RuntimeError(name+' failed with exit '+str(code))
    try:
        while True:
            f=a.diagnostics/'status.json';ds=json.loads(f.read_text()) if f.exists() else {}
            if ds.get('phase')=='completed':break
            if ds.get('phase')=='failed':raise RuntimeError('Controlled diagnostics failed: '+str(ds.get('error')))
            proc=Path('/proc')/str(a.diagnostics_pid)/'cmdline'
            if not proc.exists() or b'finish_rotation_anchor.py' not in proc.read_bytes():raise RuntimeError('Diagnostic controller exited without completion')
            save('waiting_for_controlled_validation',diagnostic_phase=ds.get('phase'));time.sleep(15)
        assert source_hash()==policy['source_sha256'] and all(sha(path)==value for path,value in bound.items())
        run('select',[sys.executable,'tools/select_rotation_candidate.py','--policy',str(a.policy),
            '--diagnostics',str(a.diagnostics),'--out',str(a.out/'frozen_selected')])
        baseline=Path(policy['baseline_checkpoint']);assert sha(baseline)==policy['baseline_sha256']
        frozen=a.out/'frozen_baseline';frozen.mkdir();destination=frozen/'baseline.pt';shutil.copy2(baseline,destination)
        assert sha(destination)==policy['baseline_sha256']
        (frozen/'freeze.json').write_text(json.dumps(dict(completed=True,name='residual_1000',checkpoint=str(destination.resolve()),
            checkpoint_sha256=policy['baseline_sha256'],selected_before_test=True,test_access_before_selection=False,
            selected_at=time.time(),policy_sha256=sha(a.policy),scope='Predeclared existing baseline, frozen before this new matched test run. Historical published internal results are not hidden.'),indent=2))
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError('Official GPU stage requires completed training/validation and idle GPUs')
        for name,gpu in [('selected','0'),('baseline','4')]:
            run('prepare_'+name,[sys.executable,'tools/prepare_standalone_bop.py','--freeze',str(a.out/('frozen_'+name)/'freeze.json'),
                '--out',str(a.out/name),'--data-root',policy['data_root'],'--index-root',policy['index_root'],
                '--initializer',policy['initializer_csv'],'--expected-initializer-sha',policy['initializer_sha256'],
                '--expected-targets-sha',policy['targets_sha256'],'--fp-root',policy['fp_root']])
            run('smoke_'+name,[sys.executable,'tools/infer_lip_tracking_bop.py','--protocol',str(a.out/name/'protocol.json'),
                '--out',str(a.out/('smoke_'+name)),'--rank','0','--world','1','--smoke-streams','2','--smoke-frames','16'],gpu)
        for name,gpus in [('selected','0,1,2,3'),('baseline','4,5,6,7')]:
            run('launch_'+name,[sys.executable,'tools/launch_standalone_bop.py','--run',str(a.out/name),'--smoke',str(a.out/('smoke_'+name)),
                '--gpus',gpus,'--workers','16','--toolkit',policy['toolkit'],'--bop-python',policy['bop_python']])
        while True:
            progress={}
            for name in ('selected','baseline'):
                folder=a.out/name;f=folder/'status.json';value=json.loads(f.read_text()) if f.exists() else {}
                if value.get('phase')=='failed':raise RuntimeError(name+' official evaluation failed: '+str(value.get('error')))
                if value.get('phase')!='completed':
                    supervisor=json.loads((folder/'supervisor.json').read_text());proc=Path('/proc')/str(supervisor['pid'])/'cmdline'
                    if not proc.exists() or b'finish_streaming_bop.py' not in proc.read_bytes():raise RuntimeError(name+' official supervisor exited early')
                progress[name]=value
            save('official_evaluation',progress=progress)
            if all(value.get('phase')=='completed' for value in progress.values()):break
            time.sleep(20)
        run('compare',[sys.executable,'tools/compare_standalone_runs.py','--selected',str(a.out/'selected'),
            '--baseline',str(a.out/'baseline'),'--out',str(a.out/'comparison')])
        assert all(sha(path)==value for path,value in bound.items()) and source_hash()==policy['source_sha256']
        save('completed',completed=True,report=str(a.out/'comparison/report.md'))
    except BaseException as error:
        if current is not None and current.poll() is None:current.terminate();current.wait()
        save('failed',error=repr(error));raise


if __name__=='__main__':main()
