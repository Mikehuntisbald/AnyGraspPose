"""Complete the prespecified rotation/startup diagnostics after matched retraining."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from run_reference_bypass_pair import upstream_status
from analyze_pose_robustness import initial_groups


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('runtime','experiment','startup-eval','original-parent-eval','hold-audit','out'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--controller-pid',required=True,type=int);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.runtime/'src'))
    from lip.engine.stream_checkpoint import source_hash
    e=json.loads((a.experiment/'experiment.json').read_text());assert e['reader_arm']=='rotation_anchor'
    helpers=('run_reference_bypass_pair.py','analyze_pose_robustness.py','audit_bad_initial_errors.py',
             'analyze_reference_feedback.py','compare_rk_ablation.py')
    bound={str(path):digest(path) for path in [Path(__file__),a.experiment/'experiment.json',
        a.runtime/'tools/run_spatial_memory.py',*[root/'tools'/n for n in helpers]]}
    status=dict(pid=os.getpid(),phase='preparing',experiment=str(a.experiment),bindings=bound,
        source_sha256=source_hash(),scope='Matched final-step val diagnostics. No candidate promotion and no test access.')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(a.runtime/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def save(phase,**extra):
        status.update(phase=phase,updated=time.time(),**extra)
        tmp=a.out/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(a.out/'status.json')
    def run(name,command):
        save(name,command=command)
        with (a.out/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        start=None;assert source_hash()==e['source_sha256']
        while True:
            phase,start=upstream_status(a.experiment,a.controller_pid,start,'run_spatial_memory.py')
            if phase=='completed':break
            save('waiting',upstream_phase=phase);time.sleep(15)
        assert source_hash()==e['source_sha256'] and all(digest(path)==value for path,value in bound.items())
        for arm in ('control','rotation_anchor'):
            receipt=json.loads((a.experiment/arm/'training_receipt.json').read_text())
            assert receipt['completed'] and receipt['stage_step']==1000
            assert digest(a.experiment/arm/'train/last.pt')==receipt['checkpoint_sha256']
        folders=dict(parent=str(a.original_parent_eval),startup_recipe=str(a.startup_eval),
            adaptive_parent=e['reference_evaluation'],control=str(a.experiment/'control/s0_val'),
            rotation_anchor=str(a.experiment/'rotation_anchor/s0_val'))
        flags=[v for name,path in folders.items() for v in ('--evaluation',name+'='+path)]
        run('robustness',[sys.executable,str(root/'tools/analyze_pose_robustness.py'),*flags,
            '--reference','control','--out',str(a.out/'robustness')])
        run('fixed_groups',[sys.executable,str(root/'tools/analyze_reference_feedback.py'),*flags,
            '--hold-audit',str(a.hold_audit),'--out',str(a.out/'fixed_groups')])
        for reference,candidate in (('control','rotation_anchor'),('adaptive_parent','control'),
                                    ('adaptive_parent','rotation_anchor'),('startup_recipe','rotation_anchor')):
            name='bad_initial_'+candidate+'_vs_'+reference
            run(name,[sys.executable,str(root/'tools/audit_bad_initial_errors.py'),
                '--reference-eval',folders[reference],'--candidate-eval',folders[candidate],
                '--index-root',e['index_root'],'--out',str(a.out/name)])
        rows=list(map(json.loads,(a.experiment/'rotation_anchor/s0_val/predictions.jsonl').read_text().splitlines()))
        initial=initial_groups(rows);stream_rows=defaultdict(list)
        for row in rows:stream_rows[row['stream_id']].append(row)
        groups=defaultdict(list)
        for sid,rs in stream_rows.items():
            for position,row in enumerate(sorted(rs,key=lambda r:r['frame_index'])[1:],1):
                value=row.get('rotation_anchor_fraction')
                if value is None:continue
                assert np.isfinite(value) and 0<=value<=1
                groups['all'].append(value)
                groups['first8' if position<=8 else 'after8'].append(value)
                groups['initial_good' if initial[sid]['good'] else 'initial_bad'].append(value)
                if not initial[sid]['good'] and position<=8:groups['initial_bad_first8'].append(value)
                visibility=row['visibility']
                if visibility is not None:groups['severe' if visibility<.3 else 'nonsevere'].append(value)
        activity={name:dict(frames=len(values),mean=float(np.mean(values)),
            positive_fraction=float(np.mean(np.asarray(values)>0)),
            quantiles_05_50_95=np.quantile(values,[.05,.5,.95]).tolist()) for name,values in groups.items()}
        (a.out/'read_activity.json').write_text(json.dumps(dict(completed=True,populations=activity,
            prediction_sha256=digest(a.experiment/'rotation_anchor/s0_val/predictions.jsonl'),
            scope='Descriptive initial-rotation read fraction, not calibrated confidence or proof of causal benefit. GT-defined groups are analysis only.'),indent=2))
        save('completed',read_activity=activity,candidate_frozen=False)
    except BaseException as error:
        save('failed',error=repr(error));raise


if __name__=='__main__':main()
