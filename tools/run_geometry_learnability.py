"""Bounded frozen-input diagnostics; no production model training."""
import json,os,subprocess,sys,time,signal,hashlib,tarfile
from pathlib import Path


def main():
    exe=Path(__file__).resolve().parents[1]
    root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_learnability_v37');root.mkdir(exist_ok=False)
    files={str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(files,indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for f in files:tar.add(exe/f,arcname=f)
    children=[];env=dict(os.environ,PYTHONPATH=str(exe/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def status(stage,**kw):
        p=root/'status.tmp';p.write_text(json.dumps(dict(stage=stage,pids=[x.pid for x in children],time=time.time(),**kw),indent=2));p.replace(root/'status.json')
    def stop(sig,_):
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(stage,commands):
        nonlocal children
        assert all(hashlib.sha256((exe/f).read_bytes()).hexdigest()==h for f,h in files.items())
        children=[];logs=[]
        try:
            for rank,args in enumerate(commands):
                log=(root/f'{stage}.{rank}.log').open('w');logs.append(log)
                children.append(subprocess.Popen([sys.executable]+args,cwd=exe,env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None,0) for p in children):raise RuntimeError(stage+' failed')
                status(stage);time.sleep(3)
            if any(p.returncode for p in children):raise RuntimeError(stage+' failed')
        finally:
            for p in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for log in logs:log.close()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        run('cache',[['tools/probe_geometry_learnability.py','cache','--out',str(root/'cache'/f'rank{i}'),'--rank',str(i)] for i in range(8)])
        arms=[('flow',1),('flow',8),('mixed',1),('mixed',8)]
        run('fit',[['tools/probe_geometry_learnability.py','fit','--cache',str(root/'cache'),'--out',str(root/f'{loss}{count}'),'--loss',loss,'--count',str(count),'--steps','400'] for loss,count in arms])
        results={f'{loss}{count}':json.loads((root/f'{loss}{count}'/'receipt.json').read_text()) for loss,count in arms}
        (root/'outcome.json').write_text(json.dumps(dict(completed=True,production_model_changed=False,goal_complete=False,arms=results),indent=2)+'\n')
        status('complete',completed=True,production_model_changed=False)
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
