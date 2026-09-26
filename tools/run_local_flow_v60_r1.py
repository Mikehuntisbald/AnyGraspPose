"""Eight-GPU frozen features, matched heads, then paired geometry probes."""
import json,os,signal,subprocess,sys,time,hashlib,tarfile
from pathlib import Path


def main():
    exe=Path(__file__).resolve().parents[1];root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/local_flow_v60_r1')
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
        run('tests',[['-m','pytest','-q','tests/jepa/test_local_flow.py','tests/jepa/test_flow_reconstruction.py','tests/jepa/test_flow_surface_audit.py']])
        (root/'cache').symlink_to('/mnt/why/dexycb_lip/unified_jepa_20260921/local_flow_v60/cache',target_is_directory=True)
        run('train2',[['tools/train_local_flow_v60.py','--root',str(root),'--stop-at','2']])
        run('train1000',[['tools/train_local_flow_v60.py','--root',str(root),'--resume','--stop-at','1000']])
        for angle,records in ((10,8),(60,4)):
            for arm in ('baseline','control','extra'):
                config='configs/jepa/flow_reconstruction_v56.yaml' if arm=='baseline' else str(root/'training'/f'{arm}.yaml')
                checkpoint='/mnt/why/dexycb_lip/unified_jepa_20260921/flow_reconstruction_v56/runs/seed42/last.pt' if arm=='baseline' else str(root/'training'/f'{arm}.pt')
                run(f'probe_{arm}_{angle}',[['tools/probe_geometry_transport.py','--config',config,'--checkpoint',checkpoint,
                    '--out',str(root/'probe'/f'{arm}_{angle}'/f'rank{i}'),'--rank',str(i),'--records',str(records),
                    '--seed-start','60050000','--rotation-degrees',str(angle)] for i in range(8)],True)
        status('complete',completed=True,head_updates=1000,backbone_frozen=True,default_model_changed=False)
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
