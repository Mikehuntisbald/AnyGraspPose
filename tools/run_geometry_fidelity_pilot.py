"""Two same-seed, same-parent continuation arms; no automatic arm selection."""
import fcntl,json,os,signal,subprocess,sys,time,tarfile
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.checkpoint import atomic_json
from lip.engine.jepa_checkpoint import sha


def main():
    exe=Path(__file__).resolve().parents[1]
    root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_fidelity_v22')
    control=root/'pilot_controller';control.mkdir(exist_ok=False)
    lock=(control/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files={str(f.relative_to(exe)):sha(f) for d in ('src','tools','configs','tests') for f in (exe/d).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    atomic_json(root/'pilot_source_receipt.json',dict(execution_root=str(exe),files=files))
    with tarfile.open(root/'pilot_source.tar.gz','w:gz') as t:
        for f in files:t.add(exe/f,arcname=f)
    env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    children=[];state=dict(completed=False,source_step=1000,target_step_per_arm=1200,arms=['control','priority'],seed=42)
    def status(job,**kw):state.update(job=job,heartbeat_unix=time.time(),**kw);atomic_json(control/'status.json',state)
    def stop(sig,_):
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(name,commands,sharded=False):
        nonlocal children
        assert all(sha(exe/f)==h for f,h in files.items()),'Pinned source changed'
        children=[];logs=[]
        try:
            for rank,args in enumerate(commands):
                f=(control/(name+f'.{rank}.log')).open('w');logs.append(f)
                children.append(subprocess.Popen([sys.executable]+args,cwd=exe,env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)) if sharded else env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
            while any(x.poll() is None for x in children):
                if any(x.poll() not in (None,0) for x in children):raise RuntimeError(name+' failed')
                status(name,pids=[x.pid for x in children]);time.sleep(5)
            if any(x.returncode for x in children):raise RuntimeError(name+' failed')
        finally:
            for x in children:
                if x.poll() is None:os.killpg(x.pid,signal.SIGTERM)
            for f in logs:f.close()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        run('tests',[['-m','pytest','-q','tests/jepa/test_geometry_priority.py','tests/jepa/test_serial_completion.py','tests/jepa/test_pose_geometry.py','tests/jepa/test_joint_pose.py','tests/jepa/test_dpt_surface.py','tests/jepa/test_rope3d.py']])
        for arm in ('control','priority'):
            config='configs/jepa/geometry_fidelity_v22_'+arm+'.yaml';c=yaml.safe_load((exe/config).read_text());out=Path(c['paths']['output']);r=out.parents[1]
            for step in (1002,1003,1200):
                args=['-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_serial_completion.py','--config',config,'--oracle',str(Path(c['serial_completion']['oracle_checkpoint']).parent),'--stop-at',str(step)]
                args+=['--resume',str(out/'last.pt')] if (out/'last.pt').exists() else ['--fidelity-from',c['geometry_priority']['source_checkpoint']]
                run(arm+'_train'+str(step),[args]);assert json.loads((out/'last.receipt.json').read_text())['step']==step
            run(arm+'_recovery',[['tools/evaluate_recovery_focus.py','--config',config,'--checkpoint',str(out/'last.pt'),'--out',str(r/'recovery/step1200')]])
            pred=r/'validation/step1200';scored=r/'validation/step1200_scored'
            run(arm+'_native',[['tools/infer_unified_jepa_val.py','--config',config,'--checkpoint',str(out/'last.pt'),'--out',str(pred/f'rank{i}'),'--rank',str(i),'--world','8','--disable-history'] for i in range(8)],True)
            run(arm+'_score',[['tools/score_unified_jepa_val.py','--run',str(pred),'--world','8','--index-root',c['paths']['index_root'],'--visibility-reference',c['paths']['visibility_reference'],'--workers','16','--out',str(scored)]])
        status('complete',completed=True)
    except Exception as error:status('failed',error=str(error));raise

if __name__=='__main__':main()
