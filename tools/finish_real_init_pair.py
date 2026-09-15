"""Finish fixed-budget adaptation with full matched val and isolated runtime."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path)
    p.add_argument('--fp-scored',required=True,type=Path);p.add_argument('--residual-scored',required=True,type=Path)
    p.add_argument('--visibility-reference',required=True,type=Path);p.add_argument('--fp-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];r=a.out.resolve();e=json.loads((a.experiment/'experiment.json').read_text())
    assert source_hash()==e['source_sha256'];r.mkdir(parents=True,exist_ok=False)
    bound={str(a.experiment/'experiment.json'):sha(a.experiment/'experiment.json'),e['val_initializers']:e['val_initializers_sha256'],e['parent']:e['parent_sha256']}
    for folder in (a.fp_scored,a.residual_scored):
        for name in ('manifest.json','predictions.jsonl'):bound[str(folder/name)]=sha(folder/name)
    for name in ('infer_lip_val_non_gt.py','score_val_non_gt.py','compare_fp_scorecard.py','benchmark_val_tracking.py','finish_real_init_pair.py'):
        bound[str(root/'tools'/name)]=sha(root/'tools'/name)
    spec=dict(bound_files=bound,args={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()},source_sha256=source_hash())
    (r/'spec.json').write_text(json.dumps(spec,indent=2));(r/'runner.lock').write_text(str(os.getpid()))
    state=dict(phase='waiting_training',started=time.time(),commands=[],models={});jobs=[]
    env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def save():
        state['updated']=time.time();f=r/'status.tmp';f.write_text(json.dumps(state,indent=2));f.replace(r/'status.json')
    def run(phase,commands):
        nonlocal jobs
        state['phase']=phase;jobs=[]
        for label,gpu,cmd,logpath in commands:
            logpath.parent.mkdir(parents=True,exist_ok=True);f=logpath.open('x')
            proc=subprocess.Popen(cmd,cwd=root,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            jobs.append((proc,f));state['commands'].append(dict(label=label,pid=proc.pid,command=cmd,log=str(logpath)));save()
        while any(p.poll() is None for p,f in jobs):
            if any(p.poll() not in (None,0) for p,f in jobs):raise RuntimeError(phase+' failed')
            save();time.sleep(5)
        for p,f in jobs:f.close();assert p.returncode==0
        jobs=[]
    def freeze(name,path,digest):
        assert sha(path)==digest;folder=r/name;folder.mkdir();dest=folder/'frozen.pt';shutil.copy2(path,dest);assert sha(dest)==digest
        obj=torch.load(dest,map_location='cpu',weights_only=False);assert obj['split_hash']==e['split_hash'] and obj['mesh_hash']==e['mesh_hash']
        cfg=folder/'config.yaml';cfg.write_text(yaml.safe_dump(obj['config'],sort_keys=False))
        state['models'][name]=dict(checkpoint=str(dest),checkpoint_sha256=digest,config=str(cfg));save()
    def infer(names,world):
        cmds=[]
        for i,n in enumerate(names):
            m=state['models'][n]
            for rank in range(world):
                out=r/n/'inference'/f'rank{rank}'
                cmd=[sys.executable,'tools/infer_lip_val_non_gt.py','--config',m['config'],'--checkpoint',m['checkpoint'],'--initializers',e['val_initializers'],
                    '--data-root',e['data_root'],'--index-root',e['index_root'],'--rank',str(rank),'--world',str(world),'--out',str(out)]
                cmds.append((n+str(rank),str(i*world+rank),cmd,out.with_suffix('.log')))
        run('inference_'+','.join(names),cmds);cmds=[]
        for n in names:
            cmd=[sys.executable,'tools/score_val_non_gt.py','--run',str(r/n/'inference'),'--world',str(world),'--index-root',e['index_root'],
                '--visibility-reference',str(a.visibility_reference),'--out',str(r/n/'scored')]
            cmds.append((n,'',cmd,r/n/'score.log'))
        run('scoring_'+','.join(names),cmds)
    try:
        save()
        while True:
            s=json.loads((a.experiment/'status.json').read_text())
            if s['phase']=='failed':raise RuntimeError('training failed: '+s.get('error','unknown'))
            if s['phase']=='completed':break
            save();time.sleep(15)
        for path,digest in bound.items():assert sha(path)==digest,path
        freeze('residual',e['parent'],e['parent_sha256'])
        for n in ('control','real_mix'):
            receipt=json.loads((a.experiment/n/'training_receipt.json').read_text())
            assert receipt['completed'] and receipt['stage_step']==1000 and receipt['rgb_unchanged'] and receipt['all_four_rank_logs_verified'] and receipt['model_tensors_finite']
            freeze(n,a.experiment/n/'train/last.pt',receipt['checkpoint_sha256']);shutil.copy2(a.experiment/n/'training_receipt.json',r/n/'training_receipt.json')
        infer(['residual'],8)
        # Source changed in training/data code only. Verify the frozen old model's
        # entire inference trajectory and scores under the new source before selection.
        old={(x['stream_id'],x['frame_index']):x for x in map(json.loads,(a.residual_scored/'predictions.jsonl').read_text().splitlines())}
        new={(x['stream_id'],x['frame_index']):x for x in map(json.loads,(r/'residual/scored/predictions.jsonl').read_text().splitlines())}
        assert new.keys()==old.keys();maximum=0.;equal=True
        for k,v in new.items():
            previous=old[k]
            for field in ('status','initialization','add_01','adds_01','adds_005'):assert v[field]==previous[field],(k,field)
            if v['pose_centered'] is not None:
                error=float(np.abs(np.asarray(v['pose_centered'])-previous['pose_centered']).max());maximum=max(maximum,error);equal &= error==0
                np.testing.assert_allclose(v['pose_centered'],previous['pose_centered'],atol=3e-5,rtol=1e-5)
            else:assert previous['pose_centered'] is None
        (r/'residual_reproduction.json').write_text(json.dumps(dict(passed=True,frames=len(new),max_pose_abs=maximum,poses_bitwise_equal=bool(equal),all_threshold_scores_identical=True,old_predictions_sha256=sha(a.residual_scored/'predictions.jsonl'),new_predictions_sha256=sha(r/'residual/scored/predictions.jsonl')),indent=2))
        infer(['control','real_mix'],4)
        cmd=[sys.executable,'tools/compare_fp_scorecard.py','--evaluation','fp='+str(a.fp_scored),'--out',str(r/'comparison')]
        for n in state['models']:cmd+=['--evaluation',n+'='+str(r/n/'scored')]
        run('comparison',[('scorecard','',cmd,r/'comparison.log')])
        comparison=json.loads((r/'comparison/comparison.json').read_text());selected=comparison['selected'];model=state['models'][selected]
        shutil.copy2(model['checkpoint'],r/'selected.pt');assert sha(r/'selected.pt')==model['checkpoint_sha256']
        # Benchmark reference and selected candidate (once if residual retained).
        fm=json.loads((a.fp_scored/'manifest.json').read_text());timings={}
        for n in dict.fromkeys(('fp','residual',selected)):
            out=r/'benchmark'/n;cmd=[sys.executable,'tools/benchmark_val_tracking.py','--method','fp' if n=='fp' else 'lip',
                '--initializers',e['val_initializers'],'--data-root',e['data_root'],'--index-root',e['index_root'],'--fp-root',str(a.fp_root),'--out',str(out)]
            if n=='fp':cmd+=['--expected-sha',fm['checkpoint_sha256']]
            else:
                m=state['models'][n];cmd+=['--config',m['config'],'--checkpoint',m['checkpoint'],'--expected-sha',m['checkpoint_sha256']]
            run('isolated_benchmark_'+n,[(n,'0',cmd,out.with_suffix('.log'))]);timings[n]=json.loads((out/'receipt.json').read_text())
        for path,digest in bound.items():assert sha(path)==digest,path
        assert source_hash()==e['source_sha256']
        receipt=dict(completed=True,selected=selected,checkpoint_sha256=sha(r/'selected.pt'),comparison_sha256=sha(r/'comparison/comparison.json'),
            models=state['models'],timing={n:t['metrics'] for n,t in timings.items()},fp_scorecard=comparison['fp_scorecard'][selected],
            test_launched=False,scope='Final1000 full real-initialized native val selection, single seed, separate pure FP baseline. No new official AR or all-metric superiority claim.')
        (r/'selection.json').write_text(json.dumps(receipt,indent=2));state.update(phase='completed',completed=time.time());save()
    except BaseException as exc:
        for p,f in jobs:
            if p.poll() is None:p.terminate()
        for p,f in jobs:p.wait();f.close()
        state.update(phase='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
