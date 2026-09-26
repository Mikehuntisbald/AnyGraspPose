"""Frozen JEPA recovery routing controls. No pose diagnostics or training."""
import hashlib,json,os,signal,subprocess,sys,time,tarfile
from pathlib import Path


def main():
    exe=Path(__file__).resolve().parents[1];base=Path('/mnt/why/dexycb_lip/unified_jepa_20260921')
    root=base/'recovery_routing_v50';root.mkdir(exist_ok=False)
    files={str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(files,indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for f in files:tar.add(exe/f,arcname=f)
    children=[]
    def status(stage,**kw):
        p=root/'status.tmp';p.write_text(json.dumps(dict(stage=stage,pids=[c.pid for c in children],time=time.time(),**kw),indent=2));p.replace(root/'status.json')
    def stop(sig,_):
        for c in children:
            if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        for arm in ('on','observed_only','rope_off'):
            assert all(hashlib.sha256((exe/f).read_bytes()).hexdigest()==v for f,v in files.items())
            children=[];logs=[]
            for rank in range(8):
                log=(root/f'{arm}.{rank}.log').open('w');logs.append(log)
                env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                cmd=[sys.executable,'tools/probe_geometry_transport.py','--config','configs/jepa/atlas_prior_v44_supervised.yaml',
                     '--checkpoint',str(base/'atlas_prior_v44/supervised/runs/seed42/last.pt'),'--out',str(root/arm/f'rank{rank}'),
                     '--rank',str(rank),'--records','8','--recovery-routing',arm]
                children.append(subprocess.Popen(cmd,cwd=exe,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError(arm+' failed')
                status(arm);time.sleep(3)
            if any(c.returncode for c in children):raise RuntimeError(arm+' failed')
            for log in logs:log.close()
        status('complete',completed=True,training=False,pose_evaluation=False,records_per_arm=64)
    except Exception as e:
        for c in children:
            if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
        status('failed',error=str(e));raise


if __name__=='__main__':main()
