"""Bounded 1000-update measured-first/recovered-later RoPE experiment; serial exclusive GPU jobs."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.jepa_checkpoint import sha
from lip.unified.checkpoint import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];c=yaml.safe_load(Path(a.config).read_text())
    out=Path(c['paths']['output']);start=c['migration']['source_step'];end=c['training']['max_steps']
    assert end-start==1000 and c['budget']['updates']==1000
    preflight=json.loads((root/'preflight/receipt.json').read_text())
    assert preflight['passed'] and preflight['config_sha256']==sha(a.config)
    assert preflight['source_sha256']==c['reconstruction_only']['source_sha256']
    assert sha(c['reconstruction_only']['source_checkpoint'])==c['reconstruction_only']['source_sha256']
    directory=root/'controller';directory.mkdir(exist_ok=True)
    lock=(directory/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files={str(f.relative_to(root)):sha(f) for folder in ('src','tools','configs','tests') for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    atomic_json(root/'runtime_receipt.json',dict(files=files,preflight_sha256=sha(root/'preflight/receipt.json')))
    state=dict(completed=False,pid=os.getpid(),source_step=start,target_step=end,mode='staged_rope_jepa_only')
    child=None
    def status(value,**kw):
        state.update(status=value,heartbeat_unix=time.time(),**kw);atomic_json(directory/'status.json',state)
    def stop(sig,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR='/tmp/dexycb_staged_rope_v18_inductor',TRITON_CACHE_DIR='/tmp/dexycb_staged_rope_v18_triton')
    def run(name,args):
        nonlocal child
        assert all(sha(root/f)==v for f,v in files.items()),'Pinned source changed'
        with (directory/(name+'.log')).open('a') as log:
            child=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:status('running',job=name,child_pid=child.pid);time.sleep(5)
            if child.returncode:raise RuntimeError(name+' failed; see job log')
    def train(step):
        args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_two_stream.py','--config',a.config,'--stop-at',str(step)]
        if (out/'last.pt').exists():args+=['--resume',str(out/'last.pt')]
        run('train_to_'+str(step),args)
        receipt=json.loads((out/'last.receipt.json').read_text());assert receipt['step']==step and receipt['all_rank_rng']==8
    def audit(name):
        run(name,[sys.executable,'tools/verify_reconstruction_only_startup.py','--config',a.config,'--out',str(root/(name+'.json'))])
    def evaluate(name,config,checkpoint):
        run(name,[sys.executable,'tools/evaluate_recovery_focus.py','--config',config,'--checkpoint',str(checkpoint),'--out',str(root/'diagnostics'/name)])
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        train(start+2);train(start+3);audit('startup_receipt')
        for offset in (500,1000):
            train(start+offset);audit('audit'+str(offset))
            (root/'milestones').mkdir(exist_ok=True)
            if offset==500:shutil.copy2(out/'last.pt',root/'milestones/update500.pt')
            evaluate('update'+str(offset),a.config,out/'last.pt')
        baseline=Path(c['validation']['source_recovery_evaluation'])
        source_report=json.loads((baseline/'summary.json').read_text())
        assert source_report['completed'] and source_report['checkpoint_sha256']==c['reconstruction_only']['source_sha256']
        assert source_report['fixed_feature_teacher']==c['validation']['fixed_feature_teacher']
        destination=root/'diagnostics/source_v17';destination.mkdir()
        for name in ('summary.json','sequence_metrics.csv','recovery_focus.png','recovery_focus.pdf'):
            shutil.copy2(baseline/name,destination/name)
        run('report',[sys.executable,'tools/report_dpt_rope3d.py','--root',str(root),'--source','source_v17','--title','V18 measured-first recovered-later RoPE'])
        run('rope_switch1000',[sys.executable,'tools/evaluate_recovery_focus.py','--config',a.config,'--checkpoint',str(out/'last.pt'),'--out',str(root/'diagnostics/rope_switch1000'),'--rope-ablation'])
        status('complete',completed=True,checkpoint_sha256=sha(out/'last.pt'))
    except Exception as error:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        status('failed',error=str(error));raise


if __name__=='__main__':main()
