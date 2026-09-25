"""Bounded training-data decoder experiment; preserves production checkpoints."""
import hashlib,json,os,signal,subprocess,sys,time,tarfile
from pathlib import Path

def main():
    exe=Path(__file__).resolve().parents[1];root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/cad_coordinate_v23_pilot')
    root.mkdir(exist_ok=False);children=[]
    config='configs/jepa/geometry_fidelity_v22_control.yaml'
    checkpoint='/mnt/why/dexycb_lip/unified_jepa_20260921/serial_completion_v21/runs/seed42/last.pt'
    files={str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(dict(runtime=str(exe),files=files),indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as t:
        for f in files:t.add(exe/f,arcname=f)
    def status(stage,**values):
        payload=dict(stage=stage,pids=[p.pid for p in children],time=time.time(),**values)
        temporary=root/'status.tmp';temporary.write_text(json.dumps(payload,indent=2));temporary.replace(root/'status.json')
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
                env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                children.append(subprocess.Popen([sys.executable]+args,cwd=exe,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
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
        run('cache',[['tools/cache_cad_coordinate_pilot.py','--config',config,'--checkpoint',checkpoint,'--out',str(root/'cache'/f'rank{rank}'),'--rank',str(rank),'--world','8','--records','32'] for rank in range(8)])
        run('fit',[['tools/fit_cad_coordinate_pilot.py','--cache',str(root/'cache'),'--checkpoint',checkpoint,'--out',str(root/arm),'--arm',arm,'--steps','500'] for arm in ('dpt','cad')])
        result={arm:json.loads((root/arm/'receipt.json').read_text()) for arm in ('dpt','cad')}
        assert all(r['completed'] for r in result.values());(root/'comparison.json').write_text(json.dumps(result,indent=2))
        status('complete',completed=True,production_changed=False)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
