"""Bounded pixel measurement-ownership experiment."""
import os,sys,subprocess,json,time,signal,hashlib,tarfile
from pathlib import Path

def main():
    exe=Path(__file__).resolve().parents[1];root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/dense_measurement_v29');root.mkdir(exist_ok=False);children=[]
    files={str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(dict(runtime=str(exe),files=files),indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for f in files:tar.add(exe/f,arcname=f)
    def status(stage,**kw):
        tmp=root/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,pids=[p.pid for p in children],time=time.time(),**kw),indent=2));tmp.replace(root/'status.json')
    def stop(sig,_):
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(stage,commands):
        nonlocal children
        children=[];logs=[]
        assert all(hashlib.sha256((exe/f).read_bytes()).hexdigest()==h for f,h in files.items())
        try:
            for rank,args in enumerate(commands):
                log=(root/(stage+f'.{rank}.log')).open('w');logs.append(log)
                children.append(subprocess.Popen([sys.executable]+args,cwd=exe,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),PYTHONPATH=str(exe/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None,0) for p in children):raise RuntimeError(stage+' failed')
                status(stage);time.sleep(5)
            if any(p.returncode for p in children):raise RuntimeError(stage+' failed')
        finally:
            for p in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for log in logs:log.close()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        run('tests',[['-m','pytest','-q','tests/jepa/test_dense_measurement.py']])
        run('cache',[['tools/cache_dense_measurement_v29.py','--config','configs/jepa/amp_preview_v25_fixed.yaml','--checkpoint','/mnt/why/dexycb_lip/unified_jepa_20260921/serial_completion_v21/runs/seed42/last.pt','--out',str(root/'cache'/f'rank{rank}'),'--rank',str(rank),'--world','8'] for rank in range(8)])
        run('fit',[['tools/fit_dense_measurement_v29.py','--cache',str(root/'cache'),'--out',str(root/'fit'),'--steps','1000']])
        result=json.loads((root/'fit/receipt.json').read_text())
        (root/'comparison.json').write_text(json.dumps(result,indent=2));status('complete',completed=True,production_changed=False)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
