"""Bounded preflight then matched four-arm training; never selects on test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from lip.engine.stream_config import config_hash,load_stream_config
from prepare_rk_ablation import ARMS


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];os.chdir(root);a.out=a.out.resolve()
    experiment=json.loads((a.out/'experiment.json').read_text())
    assert experiment['source_sha256']==source_hash()
    assert experiment['training_manifest_sha256']==sha(experiment['training_manifest'])
    # A competing launcher must not silently share or preempt another job.
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('GPUs are occupied: '+active)
    if torch.cuda.device_count()!=8:raise RuntimeError('This factorial launcher requires 8 GPUs')
    lock=os.open(a.out/'runner.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(lock,str(os.getpid()).encode());os.close(lock)
    status=dict(phase='preflight',started=time.time(),source_sha256=source_hash(),commands=[])
    def save():
        tmp=a.out/'status.json.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(a.out/'status.json')
    jobs=[]
    def parallel(phase,commands):
        nonlocal jobs
        status['phase']=phase;jobs=[]
        for arm,cmd in commands:
            rank=ARMS.index(arm);env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=f'{2*rank},{2*rank+1}',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            logpath=a.out/arm/(phase+'.log');log=logpath.open('w')
            proc=subprocess.Popen(cmd,env=env,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log));status['commands'].append(dict(arm=arm,phase=phase,command=cmd,pid=proc.pid,log=str(logpath)))
        save()
        while any(proc.poll() is None for proc,log in jobs):
            if any(proc.poll() not in (None,0) for proc,log in jobs):raise RuntimeError(phase+' child failed; inspect arm log')
            status['updated']=time.time();save();time.sleep(5)
        for proc,log in jobs:
            log.close()
            if proc.returncode:raise RuntimeError(f'{phase} child pid {proc.pid} exited {proc.returncode}; inspect arm log')
        jobs=[]
    def commands(folder,steps,resume=False,preflight=True,probe=False):
        result=[]
        for arm in ARMS:
            entry=experiment['arms'][arm];cfg=entry['config'];world=2
            if probe:
                c=load_stream_config(cfg);batch=c['batch_sequences_per_gpu']
                c.update(world_size=1,grad_accum_steps=1,effective_sequences_per_step=batch,nominal_supervised_updates_per_step=batch*c['supervised_unroll_frames'])
                cfg=str(a.out/arm/'memory_probe.yaml');Path(cfg).write_text(yaml.safe_dump(c,sort_keys=False));world=1
            dest=a.out/arm/folder
            cmd=[sys.executable]
            if world>1:cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
            cmd+=['-m','lip.train_stream','--config',cfg,'--output',str(dest),'--data-root',experiment['data_root'],'--index-root',experiment['index_root'],
                  '--max-steps',str(steps),'--fixed-manifest',experiment['training_manifest']]
            cmd+=['--resume',str(dest/'last.pt')] if resume else ['--init-from',entry['init']]
            if preflight:cmd+=['--preflight']
            result.append((arm,cmd))
        return result
    try:
        save()
        tests=['tests/test_stream_rk.py','tests/test_stream_residual_cross.py','tests/test_stream_state.py','tests/test_stream_causality.py',
               'tests/test_stream_review_regressions.py','tests/test_stream_migration.py','tests/test_stream_geometry.py','tests/test_stream_evaluation.py','tests/test_rk_analysis.py']
        env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (a.out/'tests.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(a.out/'tests.xml')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        suites=ET.parse(a.out/'tests.xml').findall('.//testsuite')
        assert sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)==0
        gpuenv=dict(env,CUDA_VISIBLE_DEVICES='0')
        with (a.out/'cuda_regression.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pytest','tests/test_stream_rk_cuda.py','-q','--junitxml='+str(a.out/'cuda_regression.xml')],
                env=gpuenv,stdout=log,stderr=subprocess.STDOUT,check=True)
        parallel('parent_equivalence',[(arm,[sys.executable,str(root/'tools/verify_rk_start.py'),'--experiment',str(a.out),'--arm',arm]) for arm in ARMS])
        parallel('memory_probe',commands('memory_probe',2,probe=True))
        parallel('ddp_probe',commands('ddp_probe',3))
        parallel('ddp_resume',commands('ddp_probe',4,resume=True))
        for arm in ARMS:
            folder=a.out/arm;entry=experiment['arms'][arm];c=load_stream_config(entry['config'])
            restored=[json.loads((folder/f'ddp_probe/resume_rank{rank}.json').read_text()) for rank in range(2)]
            assert all(r['passed'] and r['loaded_step']==3 and r['rng_restored'] for r in restored)
            rows=[json.loads(x) for x in (folder/'ddp_probe/rank0.jsonl').read_text().splitlines()]
            assert len(rows)==4 and rows[-1]['sampler_position']==256 and all(r['effective_sequences']==64 for r in rows)
            approved=dict(architecture_id=c['architecture_id'],cache_contract='lip-rk-factorial-v1',config_hash=config_hash(c),
                split_hash=experiment['split_hash'],mesh_hash=experiment['mesh_hash'],source_sha256=source_hash(),
                approved=True,completed=True,scope='Focused incremental architecture preflight; no claim of 300-step overfit',
                checks=dict(unit_tests=str(a.out/'tests.xml'),cuda_regression=str(a.out/'cuda_regression.xml'),parent_equivalence=str(folder/'equivalence.json'),
                    memory_probe=str(folder/'memory_probe/rank0.jsonl'),ddp_and_resume=str(folder/'ddp_probe'),
                    real_train_fragments=256,steps=4,world_size=2),init_sha256=entry['init_sha256'])
            (folder/'approval.json').write_text(json.dumps(approved,indent=2))
        parallel('training',commands('train',1000,preflight=False))
        for arm in ARMS:
            folder=a.out/arm;done=json.loads((folder/'train/completed.json').read_text());assert done['new_stage_step']==1000
            initial=torch.load(folder/'init.pt',map_location='cpu',weights_only=False)
            final=torch.load(folder/'train/last.pt',map_location='cpu',weights_only=False)
            trainable=experiment['arms'][arm]['trainable']
            changed=[k for k,v in initial['model'].items() if not torch.equal(v,final['model'][k])]
            assert changed and not(set(changed)-set(trainable))
            (folder/'training_receipt.json').write_text(json.dumps(dict(completed=True,step=1000,checkpoint_sha256=sha(folder/'train/last.pt'),
                changed_tensors=changed,frozen_core_unchanged=True,training_samples_sha256=experiment['training_manifest_sha256']),indent=2))
        status.update(phase='training_completed_evaluation_pending',completed=time.time());save()
        # This request is written only after the initializer protocol is chosen.
        # Absence never silently authorizes GT-derived validation.
        request=a.out/'evaluation_request.json'
        if request.exists():
            chosen=json.loads(request.read_text());initials=Path(chosen['initial_poses'])
            assert sha(initials)==chosen['initial_poses_sha256']
            status['phase']='evaluating';save()
            with (a.out/'evaluation.log').open('w') as log:
                subprocess.run([sys.executable,str(root/'tools/evaluate_rk_ablation.py'),'--out',str(a.out),'--initial-poses',str(initials)],
                    cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            status.update(phase='completed',completed=time.time());save()
    except BaseException as error:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,log in jobs:proc.wait();log.close()
        status.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
