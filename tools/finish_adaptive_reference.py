"""Analyze the active adaptive pair, then evaluate its fixed final writer on/off."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_reference_bypass_pair import upstream_status


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('runtime','experiment','original-parent-eval','hold-audit','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--controller-pid',required=True,type=int);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.runtime/'src'))
    from lip.engine.stream_checkpoint import source_hash
    e=json.loads((a.experiment/'experiment.json').read_text());assert e['reader_arm']=='adaptive_reference'
    status=dict(pid=os.getpid(),phase='preparing',commands=[],experiment=str(a.experiment),source_sha256=source_hash(),
        scope='Read-only diagnostics plus a predeclared writer intervention at the fixed final step 1000. No new training or candidate promotion, no official test.')
    bound={str(p):digest(p) for p in [Path(__file__),a.experiment/'experiment.json',a.runtime/'tools/run_spatial_memory.py',
        *[root/'tools'/name for name in ('run_reference_bypass_pair.py','evaluate_reference_bypass.py','analyze_pose_robustness.py',
            'analyze_reference_feedback.py','analyze_reference_writes.py','audit_bad_initial_errors.py','compare_rk_ablation.py')]]}
    status['bindings']=bound
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(a.runtime/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def save(phase,**extra):
        status.update(phase=phase,updated=time.time(),**extra);temp=a.out/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(a.out/'status.json')
    def run(name,command):
        status['commands'].append(dict(name=name,command=command));save(name)
        with (a.out/(name+'.log')).open('w') as log:subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        start=None;assert source_hash()==e['source_sha256']
        while True:
            phase,start=upstream_status(a.experiment,a.controller_pid,start,'run_spatial_memory.py')
            if phase=='completed':break
            save('waiting',upstream_phase=phase);time.sleep(15)
        assert source_hash()==status['source_sha256'] and all(digest(Path(path))==sha for path,sha in bound.items())
        receipt=json.loads((a.experiment/'adaptive_reference/training_receipt.json').read_text())
        assert receipt['completed'] and receipt['stage_step']==1000
        source=a.experiment/'adaptive_reference/train/last.pt';assert digest(source)==receipt['checkpoint_sha256']
        frozen=a.out/'fixed_final_1000.pt';shutil.copy2(source,frozen);assert digest(frozen)==receipt['checkpoint_sha256']
        (a.out/'frozen_checkpoint.json').write_text(json.dumps(dict(checkpoint_sha256=receipt['checkpoint_sha256'],source=str(source),
            frozen=str(frozen),stage_step=1000,promoted=False,scope='Frozen final experimental checkpoint for a predeclared same-weight intervention.'),indent=2))
        folders={'parent':str(a.original_parent_eval),'previous_candidate':e['reference_evaluation'],
            'control':str(a.experiment/'control/s0_val'),'adaptive_reference':str(a.experiment/'adaptive_reference/s0_val')}
        flags=[v for name,path in folders.items() for v in ('--evaluation',name+'='+path)]
        run('robustness',[sys.executable,str(root/'tools/analyze_pose_robustness.py'),*flags,'--reference','previous_candidate','--out',str(a.out/'robustness')])
        run('fixed_groups',[sys.executable,str(root/'tools/analyze_reference_feedback.py'),*flags,'--hold-audit',str(a.hold_audit),'--out',str(a.out/'fixed_groups')])
        for arm in ('control','adaptive_reference'):
            run('bad_initial_'+arm,[sys.executable,str(root/'tools/audit_bad_initial_errors.py'),'--reference-eval',e['reference_evaluation'],
                '--candidate-eval',str(a.experiment/arm/'s0_val'),'--index-root',e['index_root'],'--out',str(a.out/('bad_initial_'+arm))])
        run('write_dynamics',[sys.executable,str(root/'tools/analyze_reference_writes.py'),'--evaluation',str(a.experiment/'adaptive_reference/s0_val'),'--out',str(a.out/'write_dynamics')])
        run('writer_intervention',[sys.executable,str(root/'tools/run_reference_bypass_pair.py'),'--runtime',str(a.runtime),
            '--checkpoint',str(frozen),'--archived-eval',str(a.experiment/'adaptive_reference/s0_val'),
            '--config',e['arms']['adaptive_reference']['config'],'--initial-poses',e['initial_poses'],
            '--data-root',e['data_root'],'--index-root',e['index_root'],'--wait-experiment',str(a.experiment),
            '--controller-pid',str(a.controller_pid),'--controller-script','run_spatial_memory.py','--component','writer','--out',str(a.out/'writer_intervention')])
        intervention={'parent':str(a.original_parent_eval),'previous_candidate':e['reference_evaluation'],
            'adaptive_learned':str(a.out/'writer_intervention/learned'),'adaptive_writer_zero':str(a.out/'writer_intervention/zero')}
        flags=[v for name,path in intervention.items() for v in ('--evaluation',name+'='+path)]
        run('writer_fixed_groups',[sys.executable,str(root/'tools/analyze_reference_feedback.py'),*flags,'--hold-audit',str(a.hold_audit),'--out',str(a.out/'writer_fixed_groups')])
        save('completed')
    except BaseException as error:
        save('failed',error=repr(error));raise


if __name__=='__main__':main()
