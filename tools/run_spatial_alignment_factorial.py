"""Bounded preflight -> four final1000 arms -> full matched val, with receipts."""
import json,os,sys,subprocess,time
from pathlib import Path
import xml.etree.ElementTree as ET
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from lip.engine.stream_config import load_stream_config,config_hash
from lip.engine.stream_state import cache_contract_for


def main():
    root=Path(__file__).resolve().parents[1];r=root/'runs/factorial';e=json.loads((r/'experiment.json').read_text());names=list(e['arms'])
    assert e['source_sha256']==source_hash();(r/'runner.lock').open('x').write(str(os.getpid()))
    state=dict(phase='preflight',started=time.time(),commands=[]);jobs=[]
    env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def save():
        state['updated']=time.time();p=r/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(r/'status.json')
    def run(phase,commands):
        nonlocal jobs
        state['phase']=phase;jobs=[]
        for name,gpus,cmd,path in commands:
            path.parent.mkdir(parents=True,exist_ok=True);log=path.open('x')
            proc=subprocess.Popen(cmd,cwd=root,env=dict(env,CUDA_VISIBLE_DEVICES=gpus),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log));state['commands'].append(dict(phase=phase,arm=name,pid=proc.pid,command=cmd,log=str(path)));save()
        while any(p.poll() is None for p,f in jobs):
            if any(p.poll() not in (0,None) for p,f in jobs):raise RuntimeError(phase+' failed')
            save();time.sleep(5)
        for p,f in jobs:f.close();assert p.returncode==0
        jobs=[]
    def train(folder,steps,single=False,resume=False,preflight=True):
        result=[]
        for i,name in enumerate(names):
            arm=e['arms'][name];cfg=arm['config'];gpu=f'{2*i},{2*i+1}';cmd=[sys.executable]
            if single:
                c=load_stream_config(cfg);c.update(world_size=1,grad_accum_steps=1,effective_sequences_per_step=16,nominal_supervised_updates_per_step=768)
                cfg=str(r/name/'probe.yaml');Path(cfg).write_text(yaml.safe_dump(c,sort_keys=False));gpu=str(2*i)
            else:cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
            cmd+=['-m','lip.train_stream','--config',cfg,'--output',str(r/name/folder),'--max-steps',str(steps),
                '--data-root',e['data_root'],'--index-root',e['index_root'],'--fixed-manifest',e['training_manifest']]
            cmd+=['--resume',str(r/name/folder/'last.pt')] if resume else ['--init-from',arm['init']]
            if preflight:cmd+=['--preflight']
            result.append((name,gpu,cmd,r/name/(folder+('_resume' if resume else '')+'.log')))
        return result
    def infer(arms,world):
        commands=[]
        for i,(name,cfg,ckpt) in enumerate(arms):
            for rank in range(world):
                folder=r/'evaluation'/name/'inference'/f'rank{rank}'
                cmd=[sys.executable,'tools/infer_lip_val_non_gt.py','--config',cfg,'--checkpoint',ckpt,'--initializers',e['val_initializers'],
                    '--data-root',e['data_root'],'--index-root',e['index_root'],'--rank',str(rank),'--world',str(world),'--out',str(folder)]
                commands.append((name+str(rank),str(i*world+rank),cmd,folder.with_suffix('.log')))
        run('infer_'+','.join(x[0] for x in arms),commands);commands=[]
        for name,cfg,ckpt in arms:
            folder=r/'evaluation'/name
            cmd=[sys.executable,'tools/score_val_non_gt.py','--run',str(folder/'inference'),'--world',str(world),'--index-root',e['index_root'],
                '--visibility-reference','/mnt/why/dexycb_lip/adaptive_reference_20260914/runs/adaptive_pair/adaptive_reference/s0_val','--out',str(folder/'scored')]
            commands.append((name,'',cmd,folder/'score.log'))
        run('score_'+','.join(x[0] for x in arms),commands)
    try:
        save()
        for filename in ('tests.xml','cuda_tests.xml'):
            suites=list(ET.parse(root/'runs'/filename).getroot().iter('testsuite'))
            assert sum(int(s.attrib.get('errors',0))+int(s.attrib.get('failures',0)) for s in suites)==0
        equivalent=json.loads((r/'equivalence.json').read_text());assert equivalent['passed'] and equivalent['source_sha256']==e['source_sha256']
        for field in ('parent','training_manifest','train_initializers','val_initializers','models_info'):
            assert sha(e[field])==e[field+'_sha256']
        run('single_gpu_memory',train('memory_probe',2,True));run('ddp_probe',train('ddp_probe',3));run('ddp_resume',train('ddp_probe',4,resume=True))
        for name in names:
            c=load_stream_config(e['arms'][name]['config'])
            for rank in range(2):
                x=json.loads((r/name/f'ddp_probe/resume_rank{rank}.json').read_text());assert x['passed'] and x['rng_restored'] and x['loaded_step']==3
            (r/name/'approval.json').write_text(json.dumps(dict(approved=True,completed=True,architecture_id=c['architecture_id'],cache_contract=cache_contract_for(c['architecture_id']),
                config_hash=config_hash(c),source_sha256=source_hash(),split_hash=e['split_hash'],mesh_hash=e['mesh_hash'],real_preflight=str(r/'equivalence.json'),memory_and_ddp_resume_verified=True),indent=2))
        run('training',train('train',1000,preflight=False))
        for name in names:
            final=torch.load(r/name/'train/last.pt',map_location='cpu',weights_only=False)
            initial=torch.load(e['arms'][name]['init'],map_location='cpu',weights_only=False)
            assert final['new_stage_step']==final['scheduler']['last_epoch']==1000 and final['sampler_position']==64000 and len(final['rng'])==2
            assert all(torch.equal(v,final['model'][k]) for k,v in initial['model'].items() if k.startswith('rgb.'))
            assert all(torch.isfinite(v).all() for v in final['model'].values())
            total={k:0 for k in ('real_initializations_rank','real_initializations_requested_rank','real_initializations_missing_rank','primed_observations_rank')}
            for rank in range(2):
                rows=list(map(json.loads,(r/name/f'train/rank{rank}.jsonl').read_text().splitlines()));assert len(rows)==1000
                for step,row in enumerate(rows,1):
                    assert row['new_stage_step']==row['scheduler_step']==step and row['sampler_position']==64*step
                    assert row['actual_supervised_frames_rank']==1536 and row['actual_supervised_frames_global']==3072
                    assert row['primed_observations_rank']==32 and row['startup_supervised_frames_rank']==256
                    assert __import__('math').isfinite(row['loss']) and __import__('math').isfinite(row['grad_norm'])
                    for k in total:total[k]+=row[k]
            assert total['primed_observations_rank']==64000 and total['real_initializations_rank']==e['used_real_draws']
            assert total['real_initializations_requested_rank']==e['requested_real_draws'] and total['real_initializations_missing_rank']==e['missing_real_draws']
            receipt=dict(completed=True,steps=1000,checkpoint_sha256=sha(r/name/'train/last.pt'),rgb_bitwise_unchanged=True,all_rank_logs_verified=True,initialization_totals=total)
            (r/name/'training_receipt.json').write_text(json.dumps(receipt,indent=2))
            del final,initial
        # New optional code must reproduce the retained parent on all frames.
        parent=torch.load(e['parent'],map_location='cpu',weights_only=False);pc=r/'parent_eval.yaml';pc.write_text(yaml.safe_dump(parent['config'],sort_keys=False));del parent
        infer([('residual',str(pc),e['parent'])],8)
        archived=Path('/mnt/why/dexycb_lip/intraframe_alignment_20260915/runs/alignment_val/residual/scored/predictions.jsonl')
        old={(v['stream_id'],v['frame_index']):v for v in map(json.loads,archived.read_text().splitlines())}
        new={(v['stream_id'],v['frame_index']):v for v in map(json.loads,(r/'evaluation/residual/scored/predictions.jsonl').read_text().splitlines())}
        assert old.keys()==new.keys()
        for k in old:
            for field in ('pose_centered','status','add_01','adds_005'):assert old[k][field]==new[k][field]
        (r/'evaluation/parent_reproduction.json').write_text(json.dumps(dict(passed=True,frames=23200,poses_and_scores_exact=True,old_predictions_sha256=sha(archived)),indent=2))
        infer([(name,e['arms'][name]['config'],str(r/name/'train/last.pt')) for name in names],2)
        run('analysis',[('factorial','',[sys.executable,'tools/analyze_spatial_alignment_factorial.py'],r/'analysis.log')])
        # Timing runs sequentially after every GPU evaluation worker has exited.
        for name in ['residual']+names:
            cfg=str(pc) if name=='residual' else e['arms'][name]['config'];checkpoint=e['parent'] if name=='residual' else str(r/name/'train/last.pt')
            folder=r/'benchmark'/name
            cmd=[sys.executable,'tools/benchmark_val_tracking.py','--method','lip','--initializers',e['val_initializers'],'--data-root',e['data_root'],
                '--index-root',e['index_root'],'--fp-root','/mnt/why/FoundationPose','--config',cfg,'--checkpoint',checkpoint,'--expected-sha',sha(checkpoint),'--out',str(folder)]
            run('benchmark_'+name,[(name,'0',cmd,folder.with_suffix('.log'))])
        assert source_hash()==e['source_sha256'];state.update(phase='completed',completed=time.time());save()
    except BaseException as exc:
        for p,f in jobs:
            if p.poll() is None:p.terminate()
        for p,f in jobs:p.wait();f.close()
        state.update(phase='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
