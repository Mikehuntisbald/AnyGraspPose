"""Eight-GPU frozen features, matched heads, then paired geometry probes."""
import json,os,signal,subprocess,sys,time,hashlib,tarfile
from pathlib import Path


def main():
    exe=Path(__file__).resolve().parents[1];root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/official_flow_v65/experiment')
    root.mkdir(exist_ok=False);children=[]
    digest=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    files={str(p.relative_to(exe)):digest(p) for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(files,indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for f in files:tar.add(exe/f,arcname=f)
    def status(stage,**kw):
        p=root/'status.tmp';p.write_text(json.dumps(dict(stage=stage,pids=[x.pid for x in children],time=time.time(),**kw),indent=2));p.replace(root/'status.json')
    def stop(sig,_):
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(stage,commands,sharded=False):
        nonlocal children
        assert all(digest(exe/f)==h for f,h in files.items())
        children=[];logs=[]
        try:
            for rank,args in enumerate(commands):
                log=(root/f'{stage}.{rank}.log').open('w');logs.append(log)
                env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES=str(rank) if sharded else '0',
                    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                children.append(subprocess.Popen([sys.executable]+args,cwd=exe,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
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
        run('collect',[['tools/collect_official_flow_v65.py','--out',str(root/'cache'/f'rank{i}'),
            '--rank',str(i),'--records','12'] for i in range(8)],True)
        run('official',[['tools/infer_official_flow_v65.py','--rank',str(i)] for i in range(8)],True)
        status('complete',completed=True,training_performed=False,frames=96,default_model_changed=False)
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
