"""Preflight-gated, matched final1000 real-initializer adaptation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from lip.engine.stream_config import load_stream_config,config_hash
from lip.engine.stream_state import cache_contract_for


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1];r=a.experiment.resolve()
    e=json.loads((r/'experiment.json').read_text());assert source_hash()==e['source_sha256']
    for path,key in [(e['training_manifest'],'training_manifest_sha256'),(e['train_initializers'],'train_initializers_sha256'),(e['val_initializers'],'val_initializers_sha256')]:assert sha(path)==e[key]
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('Preflight requires idle GPUs')
    (r/'runner.lock').open('x').write(str(os.getpid()));state=dict(phase='starting',started=time.time(),commands=[]);jobs=[]
    def save():
        state['updated']=time.time();f=r/'status.tmp';f.write_text(json.dumps(state,indent=2));f.replace(r/'status.json')
    def run(phase,commands):
        nonlocal jobs
        state['phase']=phase;jobs=[]
        for name,gpus,cmd,path in commands:
            log=path.open('x');proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log));state['commands'].append(dict(phase=phase,arm=name,pid=proc.pid,command=cmd,log=str(path)));save()
        while any(p.poll() is None for p,f in jobs):
            if any(p.poll() not in (None,0) for p,f in jobs):raise RuntimeError(phase+' failed')
            save();time.sleep(5)
        for p,f in jobs:f.close();assert p.returncode==0
        jobs=[]
    def train(folder,steps,single=False,resume=False,preflight=True):
        commands=[]
        for i,name in enumerate(('control','real_mix')):
            entry=e['arms'][name];cfg=entry['config'];gpus=','.join(str(x) for x in range(4*i,4*i+4));cmd=[sys.executable]
            if single:
                c=load_stream_config(cfg);c.update(world_size=1,effective_sequences_per_step=16,nominal_supervised_updates_per_step=768)
                cfg=str(r/name/'probe.yaml');Path(cfg).write_text(yaml.safe_dump(c,sort_keys=False));gpus=str(4*i)
            else:cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node=4']
            cmd+=['-m','lip.train_stream','--config',cfg,'--output',str(r/name/folder),'--max-steps',str(steps),'--data-root',e['data_root'],'--index-root',e['index_root'],'--fixed-manifest',e['training_manifest']]
            cmd+=['--resume',str(r/name/folder/'last.pt')] if resume else ['--init-from',entry['init']]
            if preflight:cmd+=['--preflight']
            commands.append((name,gpus,cmd,r/name/(folder+('_resume' if resume else '')+'.log')))
        return commands
    try:
        run('tests',[('cpu','',[sys.executable,'-m','pytest','-c',str(root/'pyproject.toml'),'tests/test_real_initialization.py','tests/test_stream_startup_training.py','tests/test_stream_batch_features.py','tests/test_stream_state.py','tests/test_stream_causality.py','tests/test_stream_migration.py','tests/test_stream_sampling_contract.py','tests/test_val_non_gt_protocol.py','tests/test_stream_review_regressions.py','-m','not cuda','-q','--junitxml='+str(r/'tests.xml')],r/'tests.log')])
        run('cuda_tests',[('cuda','0',[sys.executable,'-m','pytest','-c',str(root/'pyproject.toml'),'tests/test_stream_batch_features.py','tests/test_stream_cross_cuda.py','-m','cuda','-q','--junitxml='+str(r/'cuda_tests.xml')],r/'cuda_tests.log')])
        run('real_interface_and_gradient',[('both','0',[sys.executable,'tools/verify_real_init_training.py','--experiment',str(r)],r/'equivalence.log')])
        run('memory_probe',train('memory_probe',2,single=True));run('ddp_probe',train('ddp_probe',3));run('ddp_resume',train('ddp_probe',4,resume=True))
        for name in ('control','real_mix'):
            c=load_stream_config(e['arms'][name]['config'])
            for rank in range(4):
                m=json.loads((r/name/f'ddp_probe/resume_rank{rank}.json').read_text());assert m['passed'] and m['rng_restored'] and m['loaded_step']==3
                assert m['sampling_contract']['bound_to_config']
            (r/name/'approval.json').write_text(json.dumps(dict(completed=True,approved=True,architecture_id=c['architecture_id'],cache_contract=cache_contract_for(c['architecture_id']),
                config_hash=config_hash(c),source_sha256=source_hash(),split_hash=e['split_hash'],mesh_hash=e['mesh_hash'],
                tests=str(r/'tests.xml'),real_preflight=str(r/'equivalence.json'),memory_and_ddp_resume_verified=True),indent=2))
        run('training',train('train',1000,preflight=False))
        for name in ('control','real_mix'):
            folder=r/name;final=torch.load(folder/'train/last.pt',map_location='cpu',weights_only=False);initial=torch.load(folder/'init.pt',map_location='cpu',weights_only=False)
            assert json.loads((folder/'train/completed.json').read_text())['new_stage_step']==1000
            assert final['new_stage_step']==final['scheduler']['last_epoch']==1000 and final['sampler_position']==64000 and len(final['rng'])==4
            assert final['source_sha256']==e['source_sha256']==source_hash()
            assert all(torch.equal(t,final['model'][k]) for k,t in initial['model'].items() if k.startswith('rgb.'))
            assert all(torch.isfinite(t).all() for t in final['model'].values())
            totals={k:0 for k in ('real_initializations_rank','real_initializations_requested_rank','real_initializations_missing_rank','primed_observations_rank')}
            for rank in range(4):
                rows=list(map(json.loads,(folder/f'train/rank{rank}.jsonl').read_text().splitlines()));assert len(rows)==1000
                for step,row in enumerate(rows,1):
                    assert row['new_stage_step']==row['scheduler_step']==step and row['sampler_position']==64*step
                    assert row['actual_supervised_frames_rank']==768 and row['actual_supervised_frames_global']==3072
                    assert row['primed_observations_rank']==16 and row['startup_supervised_frames_rank']==128
                    assert all(__import__('math').isfinite(v) for v in [row['loss'],row['grad_norm'],*row['metrics']])
                    for key in totals:totals[key]+=row[key]
            assert totals['primed_observations_rank']==64000
            if name=='real_mix':
                assert totals['real_initializations_rank']==e['used_real_draws'] and totals['real_initializations_requested_rank']==e['requested_real_draws']
                assert totals['real_initializations_missing_rank']==e['missing_real_draws']
            else:assert all(totals[k]==0 for k in totals if k!='primed_observations_rank')
            changed=[k for k,t in initial['model'].items() if not torch.equal(t,final['model'][k])];assert changed
            receipt=dict(completed=True,stage_step=1000,checkpoint_sha256=sha(folder/'train/last.pt'),all_four_rank_logs_verified=True,
                rgb_unchanged=True,model_tensors_finite=True,initialization_totals=totals,supervised_targets=3072000,observations=3648000,changed_tensors=changed)
            (folder/'training_receipt.json').write_text(json.dumps(receipt,indent=2))
        state.update(phase='completed',completed=time.time());save()
    except BaseException as error:
        for p,f in jobs:
            if p.poll() is None:p.terminate()
        for p,f in jobs:p.wait();f.close()
        state.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
