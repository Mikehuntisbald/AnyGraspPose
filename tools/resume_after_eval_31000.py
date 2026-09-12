"""Resume the approved run only after the complete paired evaluation succeeds."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def read_json(path):
    for attempt in range(20):
        try:return json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            if attempt==19:raise
            time.sleep(.1)

def last_record(path):
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-65536));lines=f.read().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except json.JSONDecodeError:continue
    raise RuntimeError('No complete rank log record')

def validate(comparison,manifests,pause):
    assert comparison['matched'] and comparison['checkpoint_step']==31000
    assert comparison['frames']==23200 and comparison['streams']==320
    assert pause['paused'] and pause['step']==31000
    for m in manifests:
        assert m['completed'] and m['frames']==23200 and len(m['streams'])==320
        assert m['split']=='val' and m['full_sequences'] and not m['quick_subset']
        assert m['checkpoint']['global_step']==31000
        assert m['checkpoint']['sha256']==pause['checkpoint_sha256']
    for key in ('streams','split_hash','mesh_hash','initial_pose_source'):
        assert manifests[0][key]==manifests[1][key]
    assert manifests[0]['method']=='LIP' and manifests[1]['method']=='LIP+FP'
    assert manifests[1]['history_state_source']=='post_FP' and manifests[1]['fp_iterations']==2

def main():
    R=Path('/mnt/why/dexycb_lip');os.chdir(R);J=R/'runs/lip_fp_31000'
    lock=(J/'resume.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase,**kw):
        (J/'resume_status.json').write_text(json.dumps(dict(phase=phase,utc=time.time(),**kw),indent=2))
    try:
        status('waiting_for_successful_evaluation',pid=os.getpid())
        while True:
            s=read_json(J/'status.json')
            if s['phase'].startswith('failed'):raise RuntimeError('Evaluation failed; training stays paused')
            if s['phase']=='evaluation_complete_training_paused':break
            time.sleep(5)
        validate(json.loads((J/'comparison.json').read_text()),
                 [json.loads((J/m/'manifest.json').read_text()) for m in ('lip','lip_fp')],
                 json.loads((J/'paused.json').read_text()))
        checkpoint=J/'resume_31000.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==json.loads((J/'paused.json').read_text())['checkpoint_sha256']
        import torch
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['global_step']==ck['sampler_position']==ck['scheduler']['last_epoch']==31000
        assert len(ck['rng'])==8 and ck['config']['max_optimizer_steps']==40000
        del ck
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(5)
        activation=R/'runs/basin_speed_v1/activation.json'
        (J/'training_activation_before_resume.json').write_bytes(activation.read_bytes())
        cmd=[str(R/'.venv-fp/bin/python'),str(R/'tools/run_speed_training.py'),'--resume',str(checkpoint)]
        with (J/'resumed_training_supervisor.log').open('a') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status('resume_launched',supervisor_pid=proc.pid,checkpoint=str(checkpoint),target_step=40000)
        while proc.poll() is None:
            a=read_json(activation)
            rows=[last_record(R/f'runs/lip_v1_s0/rank{r}.jsonl') for r in range(8)]
            if a.get('resume')==str(checkpoint) and (Path('/proc')/str(a['child_pid'])).exists() and all(r['run_id']=='basin_speed_v1' and r['step']>31000 and r['step']==r['scheduler_step']==r['sampler_position'] and r['nonfinite_count']==0 for r in rows):
                result=dict(phase='training_resumed_after_evaluation',utc=time.time(),supervisor_pid=proc.pid,
                            launcher_pid=a['child_pid'],checkpoint=str(checkpoint),verified_steps=[r['step'] for r in rows],target_step=40000)
                (J/'resume_status.json').write_text(json.dumps(result,indent=2))
                (J/'status.json').write_text(json.dumps(result,indent=2));return
            time.sleep(5)
        raise RuntimeError(f'Resume supervisor exited before verified progress: {proc.returncode}')
    except Exception as e:
        status('resume_failed',error=repr(e));raise

if __name__=='__main__':main()
