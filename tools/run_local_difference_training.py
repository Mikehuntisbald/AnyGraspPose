"""Bounded eight-GPU EMA continuation with an actual process-restart resume gate."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.jepa_checkpoint import sha
from lip.unified.checkpoint import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];c=yaml.safe_load(a.config.read_text())
    out=Path(c['paths']['output']);directory=root/'controller';directory.mkdir(exist_ok=True)
    lock=(directory/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=json.loads((root/'preflight/receipt.json').read_text())
    assert receipt['passed'] and receipt['config_sha256']==sha(a.config)
    assert c['training']['max_steps']==12000 and c['migration']['source_step']==11000
    assert c['budget']['updates']==1000 and c['dino_layers']==dict(student=[4,11],teacher=[4,11])
    assert c['local_structure']['scales']==[1,2,4] and c['local_structure']['local_radius']==2
    assert c['training']['loss_weights']['local_difference']==.2
    assert c['training']['loss_weights']['local_correspondence']==.05
    files={str(f.relative_to(root)):sha(f) for folder in ('src','tools','configs','tests') for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    atomic_json(root/'runtime_receipt.json',dict(files=files))
    state=dict(completed=False,pid=os.getpid(),source_step=11000,target_step=12000,mode='multiscale_local_difference_recovery_only')
    child=None
    def status(value,**kw):
        state.update(status=value,heartbeat_unix=time.time(),**kw);atomic_json(directory/'status.json',state)
    def stop(sig,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR='/tmp/dexycb_fp_v3_inductor_isolated',TRITON_CACHE_DIR='/tmp/dexycb_fp_v3_triton_isolated')
    def run(name,args):
        nonlocal child
        assert all(sha(root/f)==h for f,h in files.items()),'Pinned runtime changed'
        with (directory/(name+'.log')).open('a') as log:
            child=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                status('running',job=name,child_pid=child.pid);time.sleep(5)
            if child.returncode:raise RuntimeError(name+' failed; see controller log')
    def train(step):
        args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_two_stream.py','--config',str(a.config),'--stop-at',str(step)]
        if (out/'last.pt').exists():args+=['--resume',str(out/'last.pt')]
        run('train_to_'+str(step),args)
        saved=json.loads((out/'last.receipt.json').read_text());assert saved['step']==step and saved['all_rank_rng']==8
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs not exclusive')
        train(11002);train(11003)
        run('startup_audit',[sys.executable,'tools/verify_reconstruction_only_startup.py','--config',str(a.config)])
        run('evaluate_initial11000',[sys.executable,'tools/evaluate_recovery_focus.py','--config',str(a.config),'--checkpoint',str(out/'initial.pt'),'--out',str(root/'diagnostics/step11000')])
        train(11500)
        run('evaluate11500',[sys.executable,'tools/evaluate_recovery_focus.py','--config',str(a.config),'--checkpoint',str(out/'last.pt'),'--out',str(root/'diagnostics/step11500')])
        import shutil
        (root/'milestones').mkdir(exist_ok=True)
        shutil.copy2(out/'last.pt',root/'milestones/step11500.pt')
        train(12000)
        run('final_audit',[sys.executable,'tools/verify_reconstruction_only_startup.py','--config',str(a.config),'--out',str(root/'final_state_receipt.json')])
        run('evaluate12000',[sys.executable,'tools/evaluate_recovery_focus.py','--config',str(a.config),'--checkpoint',str(out/'last.pt'),'--out',str(root/'diagnostics/step12000')])
        run('report',[sys.executable,'tools/report_local_difference.py','--root',str(root)])
        status('complete',completed=True,checkpoint_sha256=sha(out/'last.pt'))
    except Exception as e:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        status('failed',error=str(e));raise


if __name__=='__main__':main()
