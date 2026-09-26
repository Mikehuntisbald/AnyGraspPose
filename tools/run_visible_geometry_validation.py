"""Fixed40 geometry validation of the source and both V40 terminal models."""
import json,os,subprocess,sys,time,signal,hashlib,tarfile
from pathlib import Path


def main():
    exe=Path(__file__).resolve().parents[1]
    root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/visible_correspondence_v40/full_validation');root.mkdir(exist_ok=False)
    files={str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(files,indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for f in files:tar.add(exe/f,arcname=f)
    child=None
    def status(stage,**kw):
        p=root/'status.tmp';p.write_text(json.dumps(dict(stage=stage,pid=child.pid if child else None,time=time.time(),**kw),indent=2));p.replace(root/'status.json')
    def stop(sig,_):
        if child and child.poll() is None:child.send_signal(signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    base=Path('/mnt/why/dexycb_lip/unified_jepa_20260921')
    arms=[('source','configs/jepa/geometry_surface_identity_v38_raw.yaml',base/'geometry_surface_identity_v38/raw/runs/seed42/last.pt')]
    arms += [(arm,f'configs/jepa/visible_correspondence_v40_{arm}.yaml',base/'visible_correspondence_v40'/arm/'runs/seed42/last.pt') for arm in ('corrupted','clean')]
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        for arm,config,checkpoint in arms:
            assert all(hashlib.sha256((exe/f).read_bytes()).hexdigest()==h for f,h in files.items())
            with (root/f'{arm}.log').open('w') as log:
                child=subprocess.Popen([sys.executable,'tools/evaluate_recovery_focus.py','--config',config,'--checkpoint',str(checkpoint),'--out',str(root/arm)],cwd=exe,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:status(arm);time.sleep(3)
                if child.returncode:raise RuntimeError(arm+' validation failed')
        status('complete',completed=True,physical_sequences=40,models=3,training=False)
    except Exception as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
