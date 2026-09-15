"""Matched control/spatial-memory training with real preflight and full val."""
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
from lip.engine.stream_config import load_stream_config,config_hash
from lip.engine.stream_state import cache_contract_for
from lip.evaluate_stream import merge_shards


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args();r=a.experiment.resolve();root=Path(__file__).resolve().parents[1];os.chdir(root)
    e=json.loads((r/'experiment.json').read_text());reader=e.get('reader_arm','spatial');arms=('control',reader);assert reader in ('spatial','aligned','natural','direct_pose','pose_reference','adaptive_reference','rotation_anchor','smooth_rotation');assert source_hash()==e['source_sha256'] and sha(e['training_manifest'])==e['training_manifest_sha256']
    aligned_architecture=any(load_stream_config(e['arms'][name]['config'])['architecture_id']=='stream_rk_aligned' for name in arms)
    for name in arms:
        entry=e['arms'][name]
        if 'training_manifest' in entry:assert sha(entry['training_manifest'])==entry['training_manifest_sha256']
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('Preflight requires idle GPUs')
    (r/'runner.lock').open('x').write(str(os.getpid()));status=dict(phase='preflight',commands=[],started=time.time());jobs=[]
    def save():
        tmp=r/'status.json.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(r/'status.json')
    def parallel(phase,commands):
        nonlocal jobs
        status['phase']=phase;jobs=[]
        for name,gpus,command,logpath in commands:
            env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            log=logpath.open('w');proc=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);jobs.append((proc,log))
            status['commands'].append(dict(phase=phase,arm=name,pid=proc.pid,gpus=gpus,command=command,log=str(logpath)))
        save()
        while any(p.poll() is None for p,l in jobs):
            if any(p.poll() not in (None,0) for p,l in jobs):raise RuntimeError(phase+' child failed')
            status['updated']=time.time();save();time.sleep(5)
        for proc,log in jobs:log.close();assert proc.returncode==0
        jobs=[]
    def train_commands(folder,steps,resume=False,preflight=True,single=False):
        commands=[]
        for i,name in enumerate(arms):
            entry=e['arms'][name];cfg=entry['config'];world=4;gpus=','.join(str(j) for j in range(4*i,4*i+4))
            if single:
                c=load_stream_config(cfg);c.update(world_size=1,effective_sequences_per_step=16,nominal_supervised_updates_per_step=16*48)
                cfg=str(r/name/'probe.yaml');Path(cfg).write_text(yaml.safe_dump(c,sort_keys=False));world=1;gpus=str(4*i)
            dest=r/name/folder;command=[sys.executable]
            if world>1:command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=4']
            command+=['-m','lip.train_stream','--config',cfg,'--output',str(dest),'--max-steps',str(steps),'--data-root',e['data_root'],'--index-root',e['index_root'],'--fixed-manifest',entry.get('training_manifest',e['training_manifest'])]
            command+=['--resume',str(dest/'last.pt')] if resume else ['--init-from',entry['init']]
            if preflight:command+=['--preflight']
            commands.append((name,gpus,command,r/name/(folder+('_resume' if resume else '')+'.log')))
        return commands
    try:
        save()
        env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (r/'tests.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pytest','tests/test_temporal_occlusion.py','tests/test_stream_spatial_memory.py','tests/test_stream_rk.py','tests/test_stream_state.py','tests/test_stream_causality.py','tests/test_stream_review_regressions.py','tests/test_stream_migration.py',*(['tests/test_aligned_memory.py','tests/test_stream_aligned_memory.py'] if aligned_architecture else []),*(['tests/test_natural_sampling.py','tests/test_natural_windows.py','tests/test_natural_visibility.py','tests/test_stream_sampling_contract.py'] if reader=='natural' else []),*(['tests/test_direct_pose_residual.py','tests/test_stream_direct_pose.py','tests/test_stream_sampling_contract.py'] if reader=='direct_pose' else []),*(['tests/test_pose_reference.py','tests/test_stream_pose_reference.py','tests/test_streaming_bop_protocol.py','tests/test_stream_sampling_contract.py'] if reader in ('pose_reference','adaptive_reference','rotation_anchor','smooth_rotation') else []),*(['tests/test_adaptive_reference.py','tests/test_stream_adaptive_reference.py','tests/test_stream_startup_training.py'] if reader in ('adaptive_reference','rotation_anchor','smooth_rotation') else []),*(['tests/test_rotation_anchor.py','tests/test_stream_rotation_anchor.py'] if reader in ('rotation_anchor','smooth_rotation') else []),*(['tests/test_smooth_rotation_anchor.py'] if reader=='smooth_rotation' else []),'-q','--junitxml='+str(r/'tests.xml')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if e.get('temporal_occlusion_probability',0.):
            parallel('occlusion_probe',[('control','0',[sys.executable,'tools/probe_temporal_occlusion.py','--experiment',str(r)],r/'occlusion_probe.log')])
        if aligned_architecture:
            parallel('cuda_tests',[('aligned','0',[sys.executable,'-m','pytest','tests/test_aligned_memory.py','tests/test_stream_rk_cuda.py','-m','cuda','-q','--junitxml='+str(r/'cuda_tests.xml')],r/'cuda_tests.log')])
        if reader in ('adaptive_reference','rotation_anchor','smooth_rotation'):
            parallel('cuda_tests',[('adaptive_reference','0',[sys.executable,'-m','pytest','tests/test_stream_rk_cuda.py','-q','--junitxml='+str(r/'cuda_tests.xml')],r/'cuda_tests.log')])
        verifier={'smooth_rotation':'tools/verify_smooth_rotation_start.py','rotation_anchor':'tools/verify_rotation_anchor_start.py','adaptive_reference':'tools/verify_adaptive_reference_start.py','natural':'tools/verify_natural_start.py','aligned':'tools/verify_aligned_start.py','direct_pose':'tools/verify_direct_pose_start.py','pose_reference':'tools/verify_pose_reference_start.py'}.get(reader,'tools/verify_spatial_start.py')
        parallel('equivalence',[('both','0',[sys.executable,verifier,'--experiment',str(r)],r/'equivalence.log')])
        command=[sys.executable,'-m','lip.evaluate_stream','--config',e['arms']['control']['config'],'--checkpoint',e['arms']['control']['init'],
            '--out',str(r/'reference_check'),'--data-root',e['data_root'],'--index-root',e['index_root'],'--split','val','--limit-streams','2','--initial-poses',e['initial_poses']]
        reference_commands=[('control','0',command,r/'reference_check.log')]
        reader_reference=r/(reader+'_reference_check')
        if reader in ('aligned','natural','direct_pose','pose_reference','adaptive_reference','rotation_anchor','smooth_rotation'):
            aligned_command=command.copy()
            for flag,value in [('--config',e['arms'][reader]['config']),('--checkpoint',e['arms'][reader]['init']),('--out',str(reader_reference))]:aligned_command[aligned_command.index(flag)+1]=value
            reference_commands.append((reader,'4',aligned_command,r/(reader+'_reference_check.log')))
        parallel('reference_check',reference_commands)
        old={(x['stream_id'],x['frame_index']):x for x in map(json.loads,(Path(e['reference_evaluation'])/'predictions.jsonl').read_text().splitlines())}
        new=list(map(json.loads,(r/'reference_check/predictions.jsonl').read_text().splitlines()))
        assert all(row['pose_centered']==old[(row['stream_id'],row['frame_index'])]['pose_centered'] for row in new)
        if reader in ('aligned','natural','direct_pose','pose_reference','adaptive_reference','rotation_anchor','smooth_rotation'):
            aligned_rows=list(map(json.loads,(reader_reference/'predictions.jsonl').read_text().splitlines()))
            assert len(aligned_rows)==len(new) and all(row['pose_centered']==old[(row['stream_id'],row['frame_index'])]['pose_centered'] for row in aligned_rows)
        (r/'reference_check.json').write_text(json.dumps(dict(passed=True,frames=len(new),streams=2,models=list(arms) if reader in ('aligned','natural','direct_pose','pose_reference','adaptive_reference','rotation_anchor','smooth_rotation') else ['control'],bitwise_pose_equal=True,scope='Bounded exact comparison to archived parent trajectories; not a fresh full-parent evaluation'),indent=2))
        parallel('memory_probe',train_commands('memory_probe',2,single=True))
        parallel('ddp_probe',train_commands('ddp_probe',3))
        parallel('ddp_resume',train_commands('ddp_probe',4,resume=True))
        for name in arms:
            folder=r/name;c=load_stream_config(e['arms'][name]['config'])
            resumes=[json.loads((folder/f'ddp_probe/resume_rank{rank}.json').read_text()) for rank in range(4)]
            assert all(v['passed'] and v['loaded_step']==3 and v['rng_restored'] for v in resumes)
            if reader in ('natural','direct_pose','pose_reference','adaptive_reference','rotation_anchor','smooth_rotation'):assert all(v['sampling_contract']['bound_to_config'] and v['sampling_contract']['manifest_sha256']==e['arms'][name].get('training_manifest_sha256',e['training_manifest_sha256']) for v in resumes)
            rows=[json.loads(x) for x in (folder/'ddp_probe/rank0.jsonl').read_text().splitlines()];assert len(rows)==4 and rows[-1]['sampler_position']==256
            if name==reader:
                prefix='aligned' if c['architecture_id']=='stream_rk_aligned' else 'spatial'
                assert rows[-1][prefix+'_tokens_read']>0 and rows[-1][prefix+'_update_norm']>0
                if reader=='direct_pose':assert rows[-1]['direct_rotation_norm']>0 and rows[-1]['direct_center_norm']>0
                if reader in ('pose_reference','adaptive_reference','rotation_anchor','smooth_rotation'):assert rows[-1]['reference_rotation_residual_norm']>0 and rows[-1]['reference_center_residual_norm']>0
                if reader in ('adaptive_reference','rotation_anchor','smooth_rotation'):assert rows[-1]['reference_write_rotation_norm']>0 and rows[-1]['reference_write_center_norm']>0 and rows[-1]['reference_has_training_graph']
                if reader=='rotation_anchor':assert rows[-1]['rotation_anchor_gap_norm']>0
                if reader=='smooth_rotation':assert rows[-1]['rotation_anchor_gap_norm']>0 and rows[-1]['rotation_anchor_abs_coefficient']>0
            (folder/'approval.json').write_text(json.dumps(dict(architecture_id=c['architecture_id'],cache_contract=cache_contract_for(c['architecture_id']),config_hash=config_hash(c),source_sha256=source_hash(),
                split_hash=e['split_hash'],mesh_hash=e['mesh_hash'],training_manifest_sha256=e['arms'][name].get('training_manifest_sha256',e['training_manifest_sha256']),approved=True,completed=True,checks=dict(tests=str(r/'tests.xml'),parent_equivalence=str(r/'equivalence.json'),
                    reference_pose_check=str(r/'reference_check.json'),memory_probe=str(folder/'memory_probe/rank0.jsonl'),ddp=str(folder/'ddp_probe'),resume_all_four_ranks=True)),indent=2))
        parallel('training',train_commands('train',1000,preflight=False))
        for name in arms:
            folder=r/name;done=json.loads((folder/'train/completed.json').read_text());assert done['new_stage_step']==1000
            initial=torch.load(folder/'init.pt',map_location='cpu',weights_only=False);final=torch.load(folder/'train/last.pt',map_location='cpu',weights_only=False)
            changed=[key for key,value in initial['model'].items() if not torch.equal(value,final['model'][key])]
            assert changed and not(set(changed)-set(e['arms'][name]['trainable']))
            cfg=load_stream_config(e['arms'][name]['config'])
            assert final['source_sha256']==source_hash()==e['source_sha256'] and final['config_hash']==config_hash(cfg)
            assert final['new_stage_step']==final['scheduler']['last_epoch']==1000 and final['sampler_position']==64000 and len(final['rng'])==4
            assert all(torch.isfinite(t).all().item() for t in final['model'].values())
            for rank in range(4):
                rows=list(map(json.loads,(folder/f'train/rank{rank}.jsonl').read_text().splitlines()));assert len(rows)==1000
                for expected,row in enumerate(rows,1):
                    assert row['new_stage_step']==row['scheduler_step']==expected and row['sampler_position']==64*expected
                    assert row['actual_supervised_frames_rank']==768 and row['actual_supervised_frames_global']==3072
                    assert row['startup_supervised_frames_rank']==row['omitted_late_targets_rank']==16*cfg.get('startup_supervision_frames',0)
                    assert all(__import__('math').isfinite(x) for x in [row['loss'],row['grad_norm'],*row['metrics']])
            (folder/'training_receipt.json').write_text(json.dumps(dict(completed=True,stage_step=1000,checkpoint_sha256=sha(folder/'train/last.pt'),frozen_core_unchanged=True,changed_tensors=changed,
                all_four_rank_logs_verified=True,model_tensors_finite=True,config_and_sampling_bound=True,total_supervised_targets=3072000),indent=2))
        commands=[]
        for i,name in enumerate(arms):
            folder=r/name/'s0_val';folder.mkdir(exist_ok=False)
            for rank in range(4):
                command=[sys.executable,'-m','lip.evaluate_stream','--config',e['arms'][name]['config'],'--checkpoint',str(r/name/'train/last.pt'),'--out',str(folder/f'rank{rank}'),
                    '--data-root',e['data_root'],'--index-root',e['index_root'],'--split','val','--rank',str(rank),'--world-size','4','--initial-poses',e['initial_poses']]
                commands.append((name,str(i*4+rank),command,folder/f'rank{rank}.log'))
        assert sha(e['initial_poses'])==e['initial_poses_sha256'];parallel('evaluating',commands)
        for name in arms:merge_shards(r/name/'s0_val',4)
        with (r/'comparison.log').open('w') as log:subprocess.run([sys.executable,'tools/compare_spatial_memory.py','--experiment',str(r)],stdout=log,stderr=subprocess.STDOUT,check=True)
        with (r/'paired_export.log').open('w') as log:subprocess.run([sys.executable,'tools/export_paired_stream_metrics.py','--experiment',str(r)],stdout=log,stderr=subprocess.STDOUT,check=True)
        status.update(phase='completed',completed=time.time());save()
    except BaseException as error:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,log in jobs:proc.wait();log.close()
        status.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
