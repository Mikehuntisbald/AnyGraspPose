"""Paired joint training: 500 updates per arm, serial use of all eight H20s."""
import os,sys,subprocess,json,time,signal,hashlib,tarfile,shutil
from pathlib import Path
import yaml

def main():
    exe=Path(__file__).resolve().parents[1];root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/shared_patch_joint_v32');root.mkdir(exist_ok=False)
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
    def config(arm):return 'configs/jepa/shared_patch_v32_'+arm+'.yaml'
    def train(arm,step,preflight=False):
        c=yaml.safe_load((exe/config(arm)).read_text());out=Path(c['paths']['output'])
        cmd=['-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/fp_worker.py','tools/train_serial_completion.py','--config',config(arm),'--oracle','unused_actual_prediction_trial','--stop-at',str(step)]
        if preflight:cmd+=['--preflight','--shared-patch-from',c['serial_completion']['source_checkpoint']]
        elif (out/'last.pt').exists():cmd+=['--resume',str(out/'last.pt')]
        else:cmd+=['--shared-patch-from',c['serial_completion']['source_checkpoint']]
        run(f'{arm}_'+('preflight' if preflight else 'train')+str(step),[cmd])
    def native(arm,step,off=False):
        c=yaml.safe_load((exe/config(arm)).read_text());ck=root/arm/'runs/seed42/last.pt'
        label='step'+str(step)+('_patch_off' if off else '');pred=root/arm/'validation'/label
        run(f'{arm}_{label}_native',[['tools/infer_unified_jepa_val.py','--config',config('shared_off' if off else arm),'--checkpoint',str(ck),'--out',str(pred/f'rank{i}'),'--rank',str(i),'--world','8','--disable-history'] for i in range(8)],True)
        run(f'{arm}_{label}_score',[['tools/score_unified_jepa_val.py','--run',str(pred),'--world','8','--index-root',c['paths']['index_root'],'--visibility-reference',c['paths']['visibility_reference'],'--workers','16','--out',str(pred.parent/(label+'_scored'))]])
    stopped=[]
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        run('tests',[['-m','pytest','-q','tests/jepa/test_shared_patch_joint.py','tests/jepa/test_serial_completion.py','tests/jepa/test_rope_preview_gradients.py']])
        for arm in ('control','shared'):
            train(arm,1201,True)
            for step in (1202,1203):train(arm,step)
            if arm=='shared':run('paired_start',[['tools/verify_shared_patch_v32.py','--root',str(root)]])
            train(arm,1450)
            milestone=root/arm/'milestones';milestone.mkdir(exist_ok=True)
            shutil.copy2(root/arm/'runs/seed42/last.pt',milestone/'step1450.pt')
            native(arm,1450)
            baseline=json.loads(Path('/mnt/why/dexycb_lip/unified_jepa_20260921/decoded_features_v26/validation/step1200_scored/metrics.json').read_text())['populations']['all']['adds_005']
            value=json.loads((root/arm/'validation/step1450_scored/metrics.json').read_text())['populations']['all']['adds_005']
            if value < baseline-2.0:
                stopped.append(arm)
                (root/arm/'early_stop.json').write_text(json.dumps(dict(source_adds005=baseline,step1450_adds005=value,threshold_pp=2.,completed_budget=False),indent=2))
        for arm in ('control','shared'):
            if arm in stopped:continue
            train(arm,1700);native(arm,1700)
            ck=root/arm/'runs/seed42/last.pt'
            run(arm+'_recovery',[['tools/evaluate_recovery_focus.py','--config',config(arm),'--checkpoint',str(ck),'--out',str(root/arm/'recovery/step1700')]])
            run(arm+'_controlled',[['tools/probe_serial_pose.py','--config',config(arm),'--checkpoint',str(ck),'--out',str(root/arm/'controlled/step1700'/f'rank{i}'),'--rank',str(i),'--world','8','--geometry-components'] for i in range(8)],True)
        if 'shared' not in stopped:native('shared',1700,True)
        status('complete',completed=True,source_step=1200,step=1700,updates_per_arm={arm:250 if arm in stopped else 500 for arm in ('control','shared')},early_stopped=stopped,default_model_changed=False)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
