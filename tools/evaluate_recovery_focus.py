"""Bounded frozen evaluation; this entrypoint never launches training."""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--devices',default='0,1,2,3,4,5,6,7')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];devices=a.devices.split(',')
    a.out.mkdir(parents=True,exist_ok=False);children=[];logs=[]
    def stop(sig,_):
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    snapshot={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
              for folder in ('src','tools','configs','tests') for f in (root/folder).rglob('*')
              if f.is_file() and '__pycache__' not in f.parts}
    (a.out/'source_receipt.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    try:
        for rank,device in enumerate(devices):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,PYTHONPATH=str(root/'src'),
                     OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_COMPILE_THREADS='1')
            log=(a.out/f'rank{rank}.log').open('w');logs.append(log)
            command=[sys.executable,'tools/probe_recovery_focus.py','--config',a.config,'--checkpoint',a.checkpoint,
                     '--out',str(a.out/f'rank{rank}'),'--rank',str(rank),'--world',str(len(devices))]
            children.append(subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None,0) for child in children):raise RuntimeError('Frozen probe failed; see shard logs')
            time.sleep(2)
        assert all(child.returncode==0 for child in children)
        subprocess.run([sys.executable,'tools/analyze_recovery_focus.py','--run',str(a.out),'--world',str(len(devices))],cwd=root,check=True)
        for file,digest in snapshot.items():
            assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest, 'Source changed during evaluation: '+file
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for log in logs:log.close()


if __name__=='__main__':main()
