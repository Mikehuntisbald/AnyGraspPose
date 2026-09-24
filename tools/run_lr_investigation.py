"""One paired LR replay, fixed40 selection, then continue paused progress to35400."""
import json,os,signal,subprocess,sys,time,copy,fcntl
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.jepa_checkpoint import sha
from lip.unified.checkpoint import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/staged_rope_v18_extend10k')

def main():
    directory=ROOT/'controller';directory.mkdir(exist_ok=True)
    lock=(directory/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=json.loads((OLD/'milestones/lr_investigation_pause.receipt.json').read_text())
    assert source['step']==26500 and source['sha256']=='27437eb68f6b31c145378514b36d1e48a04661f0493d5f21a5cfebad4f0b2757'
    assert sha(OLD/'milestones/lr_investigation_pause.pt')==source['sha256']
    base=yaml.safe_load((ROOT/'configs/jepa/staged_rope_v18_extend10k.yaml').read_text())
    def config(name,step,path,digest,scale):
        c=copy.deepcopy(base);c['paths']['output']=str(ROOT/'runs'/name)
        c['migration'].update(source_step=step,source_sampler_position=step*32)
        c['reconstruction_only'].update(source_checkpoint=str(path),source_sha256=digest,source_step=step)
        c['lr_intervention']=dict(scale=scale,schedule_origin=25400,rewarm_steps=200,schedule_end=35400,floor=.1)
        c['budget'].update(stage_start=step,updates=35400-step,authorization='LR-only controlled replay and continuation from preserved26500; final35400 unchanged')
        p=ROOT/'configs/jepa'/('lr_'+name+'.yaml');p.write_text(yaml.safe_dump(c,sort_keys=False));return str(p)
    probe=config('low03',25400,Path(base['reconstruction_only']['source_checkpoint']),base['reconstruction_only']['source_sha256'],.3)
    continuations={scale:config('continue_'+str(scale),26500,OLD/'milestones/lr_investigation_pause.pt',source['sha256'],scale) for scale in (.3,1.)}
    files={str(f.relative_to(ROOT)):sha(f) for folder in ('src','tools','configs','tests') for f in (ROOT/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    atomic_json(ROOT/'runtime_receipt.json',dict(files=files))
    state=dict(completed=False,pid=os.getpid(),target_step=35400,paused_step=26500,probe_scale=.3)
    child=None
    def status(value,**kw):state.update(status=value,heartbeat_unix=time.time(),**kw);atomic_json(directory/'status.json',state)
    def stop(sig,_):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR='/tmp/dexycb_staged_rope_v18_inductor',TRITON_CACHE_DIR='/tmp/dexycb_staged_rope_v18_triton')
    def run(name,args):
        nonlocal child
        assert all(sha(ROOT/f)==digest for f,digest in files.items()),'Pinned runtime changed'
        with (directory/(name+'.log')).open('a') as log:
            child=subprocess.Popen([sys.executable]+args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:status('running',job=name,child_pid=child.pid);time.sleep(5)
            if child.returncode:raise RuntimeError(name+' failed')
    def train(name,cfg,end):
        c=yaml.safe_load(Path(cfg).read_text());out=Path(c['paths']['output']);args=['-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_two_stream.py','--config',cfg,'--stop-at',str(end)]
        if (out/'last.pt').exists():args+=['--resume',str(out/'last.pt')]
        run(name,args)
        record=json.loads((out/'last.receipt.json').read_text());assert record['step']==end and record['all_rank_rng']==8
        return out/'last.pt'
    def audit(name,cfg):run(name,['tools/verify_reconstruction_only_startup.py','--config',cfg,'--out',str(ROOT/(name+'.json'))])
    def evaluate(name,cfg,checkpoint):run(name,['tools/evaluate_recovery_focus.py','--config',cfg,'--checkpoint',str(checkpoint),'--out',str(ROOT/'diagnostics'/name)])
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        train('probe_start',probe,25403);audit('probe_startup_audit',probe)
        checkpoint=train('probe_to26400',probe,26400);audit('probe_terminal_audit',probe)
        evaluate('low03_26400',probe,checkpoint)
        high=json.loads((OLD/'diagnostics/update1000/summary.json').read_text());low=json.loads((ROOT/'diagnostics/low03_26400/summary.json').read_text())
        assert high['completed'] and low['completed'] and high['physical_sequences']==low['physical_sequences']==40
        assert high['fixed_feature_teacher']==low['fixed_feature_teacher']
        rows=[]
        for case in ('light','heavy_pooled'):
            for target in ('real','proxy'):
                for metric in ('xyz_mm','depth_mm'):
                    group='geometry_focus_'+target
                    h=high['tables'][case][group]['arms']['on'][metric]['mean'];l=low['tables'][case][group]['arms']['on'][metric]['mean']
                    rows.append(dict(case=case,target=target,metric=metric,high=h,low=l,ratio=l/h))
        ratios=[x['ratio'] for x in rows]
        retrieval=[]
        for case in ('light','heavy_pooled'):
            for group in ('spatial_hidden_real','spatial_cad_proxy'):
                h=high['tables'][case][group]['arms']['on']['retrieval_top1']['mean'];l=low['tables'][case][group]['arms']['on']['retrieval_top1']['mean']
                retrieval.append(dict(case=case,group=group,high=h,low=l,delta=l-h))
        # Predetermined practical selection, not a statistical causal proof.
        acceptable=sum(ratios)/len(ratios)<=.99 and max(ratios)<=1.03 and min(x['delta'] for x in retrieval)>=-.005
        scale=.3 if acceptable else 1.
        decision=dict(selected_scale=scale,geometry=rows,retrieval=retrieval,mean_geometry_ratio=sum(ratios)/len(ratios),rule='choose0.3 only if mean geometric error>=1% better, no metric>3% worse, no retrieval>0.5pp worse',scope='single paired1000-step replay; validation used for LR selection, not independent final test',same_source_step=25400,same_sample_steps=[25400,26399],continue_from_step=26500,final_step=35400)
        atomic_json(ROOT/'decision.json',decision)
        lines=['# Paired learning-rate replay','',f'Selected scale: {scale}; restart from preserved26500, target35400.','',decision['rule'],'',decision['scope'],'','| Case | Target | Metric | Original LR | 0.3 LR |','|---|---|---|---:|---:|']
        lines += [f"| {x['case']} | {x['target']} | {x['metric']} | {x['high']:.5f} | {x['low']:.5f} |" for x in rows]
        (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        cfg=continuations[scale];state.update(selected_scale=scale,continuation_config=cfg)
        train('continuation_start',cfg,26503);audit('continuation_startup_audit',cfg)
        for step in (30400,35400):
            checkpoint=train('continue_to'+str(step),cfg,step);audit('audit'+str(step),cfg);evaluate('step'+str(step),cfg,checkpoint)
        status('complete',completed=True,checkpoint_sha256=sha(checkpoint))
    except Exception as error:
        status('failed',error=str(error));raise

if __name__=='__main__':main()
