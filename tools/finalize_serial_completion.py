"""Wait for the bounded controller, verify diagnostics and seal delivery artifacts."""
import argparse,hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    r=a.root;runtime=a.runtime;env=dict(os.environ,PYTHONPATH=str(runtime/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    while True:
        state=json.loads((r/'controller/status.json').read_text())
        if state['job']=='failed':raise RuntimeError(state)
        if state.get('completed'):break
        time.sleep(15)
    subprocess.run([sys.executable,str(Path(__file__).with_name('probe_serial_oracle_retention.py')),'--runtime',str(runtime),
        '--checkpoint',str(r/'runs/seed42/last.pt'),'--cache',str(r/'oracle_cache'),'--out',str(r/'oracle_retention1000.json')],
        cwd=runtime,env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied after controller completion')
    subprocess.run([sys.executable,str(Path(__file__).with_name('visualize_serial_completion.py')),'--config','configs/jepa/serial_completion_v21.yaml',
        '--checkpoint',str(r/'runs/seed42/last.pt'),'--cache',str(r/'oracle_cache'),'--out',str(r/'visuals')],cwd=runtime,env=dict(env,CUDA_VISIBLE_DEVICES='0'),check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('report_serial_completion.py')),'--root',str(r)],cwd=runtime,env=env,check=True)
    def sha(f):
        with f.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
    files=[]
    for folder in ('controller','controller_before_pairfix','controller_before_rehearsal','validation','pose_probe','recovery','visuals','oracle_gate','oracle_gate_v2','report_tools'):
        files.extend(f for f in (r/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    files.extend(f for f in (r/'runs/seed42').glob('*') if f.is_file() and f.suffix!='.pt')
    files.extend((r/'oracle_cache').glob('rank*/receipt.json'))
    files.extend(f for f in r.glob('*') if f.is_file() and f.suffix in ('.json','.md','.png','.pdf','.py','.yaml') and f.name!='delivery_receipt.json')
    files.extend(f for f in r.glob('source*.tar.gz') if f.is_file())
    with tarfile.open(r/'evidence.tar.gz','w:gz',compresslevel=1) as archive:
        for f in sorted(set(files)):archive.add(f,arcname=str(f.relative_to(r)))
    receipt=dict(completed=True,checkpoint_sha256=sha(r/'runs/seed42/last.pt'),evidence_sha256=sha(r/'evidence.tar.gz'),
        source_sha256=sha(r/'source_and_config.tar.gz'),files={str(f.relative_to(r)):sha(f) for f in set(files)},official_test_access=False)
    (r/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('completed','checkpoint_sha256','evidence_sha256')}))

if __name__=='__main__':main()
