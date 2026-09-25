"""Pinned V21 controller: bounded training, serialized full native validation."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil,tarfile
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.checkpoint import atomic_json
from lip.engine.jepa_checkpoint import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--adapt-from');a=p.parse_args();c=yaml.safe_load(Path(a.config).read_text())
    exe=Path(__file__).resolve().parents[1];out=Path(c['paths']['output']);root=out.parents[1]
    controller=root/'controller';controller.mkdir(exist_ok=False)
    lock=(controller/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    oracle=root/'oracle_gate_v2';gate=json.loads((oracle/'receipt.json').read_text());assert gate['passed']
    files={str(f.relative_to(exe)):sha(f) for folder in ('src','tools','configs','tests') for f in (exe/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    atomic_json(root/'runtime_receipt.json',dict(source_files=files,execution_root=str(exe),artifact_root=str(root),config_sha256=sha(a.config)))
    with tarfile.open(root/'source_and_config.tar.gz','w:gz') as tar:
        for name in files:tar.add(exe/name,arcname=name)
    env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    children=[];state=dict(completed=False,target_steps=1000,seed=42,history=False,oracle_gate_passed=True)
    def status(job,**kw):state.update(job=job,heartbeat_unix=time.time(),**kw);atomic_json(controller/'status.json',state)
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
                log=(controller/f'{name}.{rank}.log').open('w');logs.append(log)
                child_env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)) if sharded else env
                children.append(subprocess.Popen([sys.executable]+args,cwd=exe,env=child_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(x.poll() is None for x in children):
                if any(x.poll() not in (None,0) for x in children):raise RuntimeError(name+' failed; inspect controller logs')
                status(name,pids=[x.pid for x in children]);time.sleep(5)
            if any(x.returncode for x in children):raise RuntimeError(name+' failed')
        finally:
            for x in children:
                if x.poll() is None:os.killpg(x.pid,signal.SIGTERM)
            for log in logs:log.close()
    def train(step):
        nonlocal a
        if (out/'last.receipt.json').exists() and json.loads((out/'last.receipt.json').read_text())['step']>=step:return
        args=['-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_serial_completion.py','--config',a.config,'--oracle',str(oracle),'--stop-at',str(step)]
        if a.adapt_from:args+=['--adapt-from',a.adapt_from]
        elif (out/'last.pt').exists():args+=['--resume',str(out/'last.pt')]
        run('train'+str(step),[args]);assert json.loads((out/'last.receipt.json').read_text())['step']==step
        a.adapt_from=None
    def native(step,checkpoint):
        pred=root/'validation'/f'step{step}';scored=root/'validation'/f'step{step}_scored'
        if (scored/'manifest.json').exists():
            manifest=json.loads((scored/'manifest.json').read_text())
            if manifest['completed'] and manifest['checkpoint_sha256']==sha(checkpoint):return
        run('native'+str(step),[['tools/infer_unified_jepa_val.py','--config',a.config,'--checkpoint',str(checkpoint),'--out',str(pred/f'rank{r}'),'--rank',str(r),'--world','8','--disable-history'] for r in range(8)],True)
        run('score'+str(step),[['tools/score_unified_jepa_val.py','--run',str(pred),'--world','8','--index-root',c['paths']['index_root'],'--visibility-reference',c['paths']['visibility_reference'],'--workers','16','--out',str(scored)]])
        assert json.loads((scored/'manifest.json').read_text())['completed']
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        run('tests',[['-m','pytest','-q','tests/jepa/test_serial_completion.py','tests/jepa/test_pose_geometry.py','tests/jepa/test_joint_pose.py','tests/jepa/test_dpt_surface.py','tests/jepa/test_rope3d.py']])
        train(2);train(3)
        (root/'milestones').mkdir(exist_ok=True)
        for step in (500,1000):
            train(step);checkpoint=root/'milestones'/f'step{step}.pt'
            if not checkpoint.exists():
                assert json.loads((out/'last.receipt.json').read_text())['step']==step
                shutil.copy2(out/'last.pt',checkpoint)
            native(step,checkpoint)
            receipts=[root/'pose_probe'/f'step{step}'/f'rank{r}'/'receipt.json' for r in range(8)]
            if not all(p.exists() and json.loads(p.read_text())['completed'] and json.loads(p.read_text())['checkpoint_sha256']==sha(checkpoint) for p in receipts):
                run('pose_probe'+str(step),[['tools/probe_serial_pose.py','--config',a.config,'--checkpoint',str(checkpoint),'--out',str(root/'pose_probe'/f'step{step}'/f'rank{r}'),'--rank',str(r),'--world','8'] for r in range(8)],True)
        run('recovery1000',[['tools/evaluate_recovery_focus.py','--config',a.config,'--checkpoint',str(out/'last.pt'),'--out',str(root/'recovery/step1000')]])
        status('complete',completed=True,checkpoint_sha256=sha(out/'last.pt'))
    except Exception as error:status('failed',error=str(error));raise

if __name__=='__main__':main()
