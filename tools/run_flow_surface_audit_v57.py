"""Freeze V56; audit endpoint information before changing recovery decoding."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,tarfile
from pathlib import Path


def main():
    exe=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser();p.add_argument('--root',default='/mnt/why/dexycb_lip/unified_jepa_20260921/flow_surface_audit_v57')
    args=p.parse_args();root=Path(args.root);root.mkdir(exist_ok=False)
    checkpoint=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/flow_reconstruction_v56/runs/seed42/last.pt')
    expected='044a8f928d324e54c8e2cab2927dd31e55942cde98bc09fa5f18ce5245d0fa5f'
    digest=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    assert digest(checkpoint)==expected
    files={str(p.relative_to(exe)):digest(p) for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(files,indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for f in files:tar.add(exe/f,arcname=f)
    children=[]
    def status(stage,**kw):
        p=root/'status.tmp';p.write_text(json.dumps(dict(stage=stage,pids=[x.pid for x in children],time=time.time(),**kw),indent=2));p.replace(root/'status.json')
    def stop(sig,_):
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        for angle,records in ((10,8),(60,4)):
            assert all(digest(exe/f)==h for f,h in files.items())
            logs=[];children=[]
            try:
                for rank in range(8):
                    log=(root/f'angle{angle}.rank{rank}.log').open('w');logs.append(log)
                    command=[sys.executable,'tools/probe_geometry_transport.py','--config','configs/jepa/flow_reconstruction_v56.yaml',
                        '--checkpoint',str(checkpoint),'--out',str(root/f'angle{angle}'/f'rank{rank}'),
                        '--rank',str(rank),'--records',str(records),'--seed-start','56050000',
                        '--rotation-degrees',str(angle),'--flow-surface-audit']
                    env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    children.append(subprocess.Popen(command,cwd=exe,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
                while any(p.poll() is None for p in children):
                    if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Shard failed')
                    status(f'angle{angle}');time.sleep(3)
                if any(p.returncode for p in children):raise RuntimeError('Shard failed')
            finally:
                for child in children:
                    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                for log in logs:log.close()
        assert digest(checkpoint)==expected
        status('complete',completed=True,checkpoint_sha256=expected,weights_unchanged=True,training=False,
               candidate_uses_teacher=False,oracle_controls_separate=True,pose_training=False,pose_evaluation=False,
               scope='Frozen geometry information audit. Rigid fit uses PnP but reports reconstructed geometry only; not a new inference architecture.')
    except Exception as exc:status('failed',error=str(exc));raise


if __name__=='__main__':main()
