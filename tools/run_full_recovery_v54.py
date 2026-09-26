"""Exclusive8-GPU paired full s0-val recovery; tests, smoke, full, report."""
import hashlib,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path


def main():
    exe=Path(__file__).resolve().parents[1]
    root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/recovery_fullval_v54_r1');root.mkdir(exist_ok=False)
    files={str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src','tools','configs','tests')
        for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(files,indent=2)+'\n')
    with tarfile.open(root/'source.tar.gz','w:gz') as archive:
        for f in files:archive.add(exe/f,arcname=f)
    children=[]
    def status(stage,**extra):
        p=root/'status.tmp';p.write_text(json.dumps(dict(stage=stage,pids=[c.pid for c in children],time=time.time(),**extra),indent=2));p.replace(root/'status.json')
    def stop(sig,_):
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(stage,commands,sharded=False):
        nonlocal children
        assert all(hashlib.sha256((exe/f).read_bytes()).hexdigest()==h for f,h in files.items())
        children=[];logs=[]
        try:
            for rank,command in enumerate(commands):
                log=(root/f'{stage}.{rank}.log').open('w');logs.append(log)
                env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES=str(rank) if sharded else '0,1,2,3,4,5,6,7',
                    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                children.append(subprocess.Popen([sys.executable]+command,cwd=exe,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError(stage+' failed; see shard logs')
                status(stage);time.sleep(3)
            if any(c.returncode for c in children):raise RuntimeError(stage+' failed')
        finally:
            for c in children:
                if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
            for log in logs:log.close()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        run('tests',[['-m','pytest','-q','tests/jepa/test_full_recovery_metrics.py','tests/jepa/test_canonical_surface_targets.py']])
        for phase in ('smoke','full'):
            commands=[['tools/probe_full_recovery_v54.py','--out',str(root/phase/f'rank{i}'),'--rank',str(i)]+(['--smoke'] if phase=='smoke' else []) for i in range(8)]
            run(phase,commands,True)
            for i in range(8):
                m=json.loads((root/phase/f'rank{i}/manifest.json').read_text())
                assert m['completed'] and m['paired_raw_inputs_exact'] and m['targets_built_once_per_pair']
                if phase=='smoke':assert m['evaluated_frames']==2
        run('report',[['tools/report_full_recovery_v54.py','--root',str(root/'full')]])
        status('complete',completed=True,native_frames=23200,streams=320,physical_sequences=40,paired_conditions=69600,models=2,training=False,pose_evaluation=False,default_model_changed=False)
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
