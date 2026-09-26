"""100-update paired pilot; test, resume, evaluate, stop and report."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import hashlib
import tarfile


def main():
    exe = Path(__file__).resolve().parents[1]
    root = Path('/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_transport_v34')
    root.mkdir(exist_ok=False)
    files = {str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('src','tools','configs','tests') for p in (exe/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(files,indent=2)+'\n')
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for path in files: tar.add(exe/path,arcname=path)
    children = []; env = dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def status(stage,**kw):
        path=root/'status.tmp';path.write_text(json.dumps(dict(stage=stage,pids=[c.pid for c in children],time=time.time(),**kw),indent=2));path.replace(root/'status.json')
    def stop(sig,_):
        for c in children:
            if c.poll() is None: os.killpg(c.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(stage,commands,sharded=False):
        nonlocal children
        assert all(hashlib.sha256((exe/p).read_bytes()).hexdigest()==digest for p,digest in files.items()),'Pinned source changed'
        children=[];logs=[]
        try:
            for rank,args in enumerate(commands):
                log=(root/f'{stage}.{rank}.log').open('w');logs.append(log)
                children.append(subprocess.Popen([sys.executable]+args,cwd=exe,env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)) if sharded else env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError(stage+' failed')
                status(stage);time.sleep(3)
            if any(c.returncode for c in children):raise RuntimeError(stage+' failed')
        finally:
            for c in children:
                if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
            for log in logs:log.close()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        run('tests',[['-m','pytest','-q','tests/jepa/test_cad_transport.py','tests/jepa/test_dpt_surface.py','tests/jepa/test_serial_completion.py','tests/jepa/test_rope_preview_gradients.py']])
        for arm in ('control','transport'):
            config=f'configs/jepa/geometry_transport_v34_{arm}.yaml';out=root/arm/'runs/seed42'
            def train(step,resume=False):
                args=['-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_geometry_transport.py','--config',config,'--stop-at',str(step)]
                if resume:args+=['--resume',str(out/'last.pt')]
                run(arm+f'_train{step}',[args])
            def probe(step):
                ck=out/('initial.pt' if step==0 else 'last.pt')
                run(arm+f'_probe{step}',[['tools/probe_geometry_transport.py','--config',config,'--checkpoint',str(ck),'--out',str(root/arm/'probe'/f'step{step}'/f'rank{i}'),'--rank',str(i),'--world','8','--records','8'] for i in range(8)],True)
            train(2);probe(0);train(3,True);train(100,True);probe(100)
        run('report',[['tools/report_geometry_transport.py','--root',str(root)]])
        status('complete',completed=True,updates_per_arm=100,default_model_changed=False)
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
