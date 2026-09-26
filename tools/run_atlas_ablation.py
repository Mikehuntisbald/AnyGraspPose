"""Frozen two-arm retrieval intervention; no teacher data in either search."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=['prior_only','learned_only'],default=['prior_only','learned_only']);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=False)
    snapshot={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('src','tools','configs','tests') for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    (a.out/'source_receipt.json').write_text(json.dumps(snapshot,indent=2))
    children=[]
    def status(stage,**kw):
        p=a.out/'status.tmp';p.write_text(json.dumps(dict(stage=stage,pids=[x.pid for x in children],time=time.time(),**kw),indent=2));p.replace(a.out/'status.json')
    def stop(sig,_):
        for c in children:
            if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        for arm in a.arms:
            assert all(hashlib.sha256((root/f).read_bytes()).hexdigest()==s for f,s in snapshot.items())
            logs=[];children=[]
            for rank in range(8):
                log=(a.out/f'{arm}.{rank}.log').open('w');logs.append(log)
                env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                command=[sys.executable,'tools/probe_geometry_transport.py','--config',a.config,'--checkpoint',a.checkpoint,'--out',str(a.out/arm/'probe/step100'/f'rank{rank}'),'--rank',str(rank),'--world','8','--records','8','--atlas-ablation',arm]
                children.append(subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError(arm+' failed')
                status(arm);time.sleep(3)
            if any(c.returncode for c in children):raise RuntimeError(arm+' failed')
            for log in logs:log.close()
        status('complete',completed=True,training=False)
    except Exception as e:
        for c in children:
            if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
        status('failed',error=str(e));raise


if __name__=='__main__':main()
