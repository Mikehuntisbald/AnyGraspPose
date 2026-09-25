"""500 restoration updates through a frozen, independently validated pose readout."""
import fcntl,json,os,signal,subprocess,sys,time,tarfile,hashlib
from pathlib import Path
import yaml

def main():
    exe=Path(__file__).resolve().parents[1];config='configs/jepa/shape_conditioned_v24_recovery.yaml';c=yaml.safe_load((exe/config).read_text())
    out=Path(c['paths']['output']);root=out.parents[1];controller=root/'controller';controller.mkdir(parents=True,exist_ok=False)
    lock=(controller/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files={str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(dict(runtime=str(exe),files=files),indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for f in files:tar.add(exe/f,arcname=f)
    env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    children=[]
    def status(stage,**extra):
        temp=controller/'status.tmp';temp.write_text(json.dumps(dict(stage=stage,pids=[p.pid for p in children],time=time.time(),**extra),indent=2));temp.replace(controller/'status.json')
    def stop(sig,_):
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(stage,commands,sharded=False):
        nonlocal children
        assert all(hashlib.sha256((exe/f).read_bytes()).hexdigest()==h for f,h in files.items()),'Pinned source changed'
        children=[];logs=[]
        try:
            for rank,args in enumerate(commands):
                log=(controller/(stage+f'.{rank}.log')).open('w');logs.append(log)
                children.append(subprocess.Popen([sys.executable]+args,cwd=exe,env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)) if sharded else env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None,0) for p in children):raise RuntimeError(stage+' failed')
                status(stage);time.sleep(5)
            if any(p.returncode for p in children):raise RuntimeError(stage+' failed')
        finally:
            for p in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for log in logs:log.close()
    def train(step,preflight=False):
        args=['-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_serial_completion.py','--config',config,'--oracle',str(Path(c['serial_completion']['oracle_checkpoint']).parent),'--stop-at',str(step)]
        if preflight:args+=['--preflight','--readout-adapt-from',c['serial_completion']['source_checkpoint']]
        elif (out/'last.pt').exists():args+=['--resume',str(out/'last.pt')]
        else:args+=['--readout-adapt-from',c['serial_completion']['source_checkpoint']]
        run(('preflight' if preflight else 'train')+str(step),[args])
    def evaluate(step):
        pred=root/'validation'/f'step{step}'
        run(f'native{step}',[['tools/infer_unified_jepa_val.py','--config',config,'--checkpoint',str(out/'last.pt'),'--out',str(pred/f'rank{i}'),'--rank',str(i),'--world','8','--disable-history'] for i in range(8)],True)
        run(f'score{step}',[['tools/score_unified_jepa_val.py','--run',str(pred),'--world','8','--index-root',c['paths']['index_root'],'--visibility-reference',c['paths']['visibility_reference'],'--workers','16','--out',str(pred.parent/(pred.name+'_scored'))]])
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        train(1001,True)
        for step in (1002,1003,1250):train(step)
        evaluate(1250)
        # The declared stage ends at1500; do not expand budget or promote weights.
        train(1500);evaluate(1500)
        run('recovery1500',[['tools/evaluate_recovery_focus.py','--config',config,'--checkpoint',str(out/'last.pt'),'--out',str(root/'recovery/step1500')]])
        run('controlled1500',[['tools/probe_serial_pose.py','--config',config,'--checkpoint',str(out/'last.pt'),'--out',str(root/'controlled/step1500'/f'rank{i}'),'--rank',str(i),'--world','8','--geometry-components'] for i in range(8)],True)
        status('complete',completed=True,step=1500)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
