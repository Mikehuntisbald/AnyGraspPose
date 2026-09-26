"""Finish the failed same-weight intervention without restarting training."""
import os,sys,subprocess,json,time,signal,hashlib,tarfile,shutil
from pathlib import Path
import yaml

def main():
    exe=Path(__file__).resolve().parents[1];root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/shared_patch_joint_v32');root=root/"finish";root.mkdir(exist_ok=False)
    files={str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(dict(runtime=str(exe),files=files),indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for f in files:tar.add(exe/f,arcname=f)
    children=[]
    env=dict(os.environ,PYTHONPATH=str(exe/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def status(stage,**kw):
        tmp=root/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,pids=[p.pid for p in children],time=time.time(),**kw),indent=2));tmp.replace(root/'status.json')
    def stop(sig,_):
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(stage,commands,sharded=False):
        nonlocal children
        assert all(hashlib.sha256((exe/f).read_bytes()).hexdigest()==h for f,h in files.items()),'Pinned source changed'
        children=[];logs=[]
        try:
            for rank,args in enumerate(commands):
                log=(root/(stage+f'.{rank}.log')).open('w');logs.append(log)
                children.append(subprocess.Popen([sys.executable]+args,cwd=exe,env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)) if sharded else env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None,0) for p in children):raise RuntimeError(stage+' failed')
                status(stage);time.sleep(5)
            if any(p.returncode for p in children):raise RuntimeError(stage+' failed')
        finally:
            for p in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for log in logs:log.close()
    try:
        old=json.loads((root.parent/'status.json').read_text())
        if old['stage']!='failed' or old.get('error')!='shared_step1700_patch_off_native failed':raise ValueError('Unexpected recovery state')
        for pid in old['pids']:
            p=Path(f'/proc/{pid}/cmdline')
            if p.exists() and b'dexycb' in p.read_bytes():raise RuntimeError('Original inference still active')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        config='configs/jepa/shared_patch_v32_shared.yaml';c=yaml.safe_load((exe/config).read_text())
        ck=root.parent/'shared/runs/seed42/last.pt';pred=root.parent/'shared/validation/step1700_patch_off_verified'
        run('patch_off_native',[['tools/infer_unified_jepa_val.py','--config',config,'--checkpoint',str(ck),'--out',str(pred/f'rank{i}'),'--rank',str(i),'--world','8','--disable-history','--disable-shared-patch'] for i in range(8)],True)
        run('patch_off_score',[['tools/score_unified_jepa_val.py','--run',str(pred),'--world','8','--index-root',c['paths']['index_root'],'--visibility-reference',c['paths']['visibility_reference'],'--workers','16','--out',str(pred.parent/(pred.name+'_scored'))]])
        for arm in ('control','shared'):
            run(arm+'_controlled_report',[['tools/report_serial_geometry_probe.py','--root',str(root.parent/arm/'controlled/step1700')]])
        status('complete',completed=True,training_restarted=False,checkpoint_config_unchanged=True,intervention='shared patch scale zero only')
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
