"""Sealed full-val screening; fixed final1000 candidates, no test launch."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--spec',required=True,type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];spec=json.loads(a.spec.read_text());out=Path(spec['out'])
    assert source_hash()==spec['source_sha256']
    for path,digest in spec['bound_files'].items():assert sha(path)==digest,path
    out.mkdir(parents=True,exist_ok=False);(out/'runner.lock').write_text(str(os.getpid()))
    state=dict(phase='starting',started=time.time(),spec_sha256=sha(a.spec),commands=[],models={});jobs=[]
    env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def save():
        state['updated']=time.time();tmp=out/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'status.json')
    def parallel(phase,commands):
        nonlocal jobs
        state['phase']=phase;jobs=[]
        for label,gpu,command,logpath in commands:
            logpath.parent.mkdir(parents=True,exist_ok=True);log=logpath.open('x')
            proc=subprocess.Popen(command,cwd=root,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log));state['commands'].append(dict(label=label,pid=proc.pid,gpu=gpu,command=command,log=str(logpath)))
        save()
        while any(proc.poll() is None for proc,log in jobs):
            if any(proc.poll() not in (None,0) for proc,log in jobs):raise RuntimeError(phase+' child failed')
            save();time.sleep(5)
        for proc,log in jobs:log.close();assert proc.returncode==0
        jobs=[]
    def freeze(name,path,expected):
        assert sha(path)==expected
        folder=out/name;folder.mkdir();dest=folder/'frozen.pt';shutil.copyfile(path,dest);assert sha(dest)==expected
        obj=torch.load(dest,map_location='cpu',weights_only=False);assert obj['split_hash']==spec['split_hash'] and obj['mesh_hash']==spec['mesh_hash']
        config=folder/'config.yaml';config.write_text(yaml.safe_dump(obj['config'],sort_keys=False))
        state['models'][name]=dict(checkpoint=str(dest),checkpoint_sha256=expected,config=str(config),config_sha256=sha(config))
        save()
    def infer(names,world):
        commands=[]
        for i,name in enumerate(names):
            model=state['models'][name];folder=out/name/'inference'
            for rank in range(world):
                command=[sys.executable,'tools/infer_lip_val_non_gt.py','--config',model['config'],'--checkpoint',model['checkpoint'],
                    '--initializers',spec['initializers'],'--data-root',spec['data_root'],'--index-root',spec['index_root'],
                    '--rank',str(rank),'--world',str(world),'--out',str(folder/f'rank{rank}')]
                commands.append((name+f'/rank{rank}',str(i*world+rank),command,folder/f'rank{rank}.log'))
        parallel('inference:'+','.join(names),commands)
    def score(names,world):
        commands=[]
        for name in names:
            folder=out/name
            cmd=[sys.executable,'tools/score_val_non_gt.py','--run',str(folder/'inference'),'--world',str(world),
                '--index-root',spec['index_root'],'--visibility-reference',spec['visibility_reference'],'--out',str(folder/'scored')]
            commands.append((name,'',cmd,folder/'scoring.log'))
        parallel('scoring:'+','.join(names),commands)
    try:
        save()
        for name,entry in spec['fixed_models'].items():
            freeze(name,entry['checkpoint'],entry['checkpoint_sha256']);infer([name],8);score([name],8)
        training=Path(spec['training_experiment']);state['phase']='waiting_final1000_training_receipts';save()
        while True:
            status=json.loads((training/'status.json').read_text())
            if status['phase']=='failed':raise RuntimeError('training failed: '+status.get('error','unknown'))
            if all((training/name/'training_receipt.json').is_file() for name in ('control','smooth_rotation')):break
            save();time.sleep(15)
        for name in ('control','smooth_rotation'):
            receipt=json.loads((training/name/'training_receipt.json').read_text())
            assert receipt['completed'] and receipt['stage_step']==1000 and receipt['frozen_core_unchanged']
            assert receipt['all_four_rank_logs_verified'] and receipt['model_tensors_finite'] and receipt['config_and_sampling_bound']
            freeze(name,training/name/'train/last.pt',receipt['checkpoint_sha256'])
            shutil.copyfile(training/name/'training_receipt.json',out/name/'training_receipt.json')
        infer(['control','smooth_rotation'],4);score(['control','smooth_rotation'],4)
        for path,digest in spec['bound_files'].items():assert sha(path)==digest,path
        assert source_hash()==spec['source_sha256']
        cmd=[sys.executable,'tools/compare_val_non_gt.py','--out',str(out/'comparison')]
        for name in state['models']:cmd+=['--evaluation',name+'='+str(out/name/'scored')]
        parallel('comparison',[('comparison','',cmd,out/'comparison.log')])
        selection=json.loads((out/'comparison/comparison.json').read_text())['selection']
        selected=state['models'][selection['selected']];dest=out/'selected.pt';shutil.copyfile(selected['checkpoint'],dest)
        assert sha(dest)==selected['checkpoint_sha256']
        receipt=dict(completed=True,selected=selection['selected'],checkpoint=str(dest),checkpoint_sha256=sha(dest),
            selection=selection,spec_sha256=sha(a.spec),comparison_sha256=sha(out/'comparison/comparison.json'),
            test_launched=False,scope='Candidate frozen after full real non-GT val screening. Single seed; no SOTA or new test claim.')
        (out/'selection.json').write_text(json.dumps(receipt,indent=2));state.update(phase='completed',completed=time.time());save()
    except BaseException as exc:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,log in jobs:proc.wait();log.close()
        state.update(phase='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
