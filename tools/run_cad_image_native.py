"""Exclusive eight-GPU fixed40 conditional native evaluation, three frozen models."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,tarfile
from pathlib import Path


def main():
    exe=Path(__file__).resolve().parents[1];base=Path('/mnt/why/dexycb_lip/unified_jepa_20260921')
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v46','v47'],default='v46');args=parser.parse_args()
    experiment='cad_image_'+args.version
    if not json.loads((base/experiment/'status.json').read_text()).get('completed'):raise RuntimeError('Training/probes must complete first')
    root=base/experiment/'native40';root.mkdir(exist_ok=False)
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
    arms=[('source','configs/jepa/geometry_surface_identity_v38_raw.yaml',base/'geometry_surface_identity_v38/raw/runs/seed42/last.pt'),
          ('parent','configs/jepa/cad_image_v45.yaml',base/'cad_image_v45/runs/seed42/initial.pt'),
          ('trained','configs/jepa/cad_image_v46.yaml',base/'cad_image_v46/runs/seed42/last.pt')]
    if args.version=='v47':
        arms=[(name,'configs/jepa/cad_image_v47.yaml',base/experiment/'runs/seed42'/file) for name,file in [('initial','initial.pt'),('trained','last.pt')]]
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        for arm,config,checkpoint in arms:
            assert all(hashlib.sha256((exe/f).read_bytes()).hexdigest()==v for f,v in files.items())
            children=[];logs=[]
            for rank in range(8):
                log=(root/f'{arm}.{rank}.log').open('w');logs.append(log)
                env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                command=[sys.executable,'tools/probe_cad_image_native.py','--config',config,'--checkpoint',str(checkpoint),'--out',str(root/arm/f'rank{rank}'),'--rank',str(rank)]
                children.append(subprocess.Popen(command,cwd=exe,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError(arm+' failed')
                status(arm);time.sleep(3)
            if any(c.returncode for c in children):raise RuntimeError(arm+' failed')
            for log in logs:log.close()
        status('complete',completed=True,physical_sequences=40,training=False,default_model_changed=False)
    except Exception as e:
        for c in children:
            if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
        status('failed',error=str(e));raise


if __name__=='__main__':main()
