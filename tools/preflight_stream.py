"""Bounded streaming preflight; never starts a long run or preempts another GPU job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.config import check_data_gate,environment
from lip.engine.stream_config import load_stream_config,make_model,config_hash
from lip.engine.stream_checkpoint import load_init,sha,source_hash
from lip.engine.stream_state import cache_contract_for
from lip.engine.stream_training import StreamTrainingModule
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer


def occupied_gpus():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
    return [line for line in output.splitlines() if line.strip()]


def fragment_metrics(config,checkpoint,root,index,manifest,out):
    torch.set_num_threads(config['cpu_threads']);torch.cuda.set_device(0)
    ds=StreamClips(root,index,config['burn_in_frames'],config['supervised_unroll_frames'],fixed=manifest,decode_threads=config['decode_threads'])
    model=make_model(config).cuda().eval();load_init(checkpoint,model,check_data_gate(index),config)
    runner=StreamTrainingModule(model,config,Renderer('cuda'));rows=[]
    with torch.no_grad():
        for i in range(len(manifest)):
            sample=ds[i];result=runner([sample]);rows.append(dict(sample=i,loss=float(result['loss']),metrics=result['metrics'].cpu().tolist(),stream_id=sample['stream']['stream_id']))
    report=dict(completed=True,checkpoint_sha256=sha(checkpoint),rows=rows,mean_loss=float(np.mean([r['loss'] for r in rows])),
                mean_metrics=np.mean([r['metrics'] for r in rows],axis=0).tolist(),scope='Fixed training fragments, known noisy initialization and own autoregressive history; not validation accuracy')
    Path(out).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


def geometry_check(root,index,manifest,out):
    import cv2
    from lip.engine.stream_features import mesh_to_device
    ds=StreamClips(root,index,fixed=manifest);renderer=Renderer('cuda');rows=[]
    for i in range(len(manifest)):
        s=ds[i];mesh=mesh_to_device(s['mesh'],'cuda');gt=s['targets'][0].cuda();K=s['k'].cuda()
        d,_=renderer(mesh,gt,K,640);d=d[0,:480].cpu().numpy()
        observed=s['depth'][0,0].numpy().astype('f4')*s['depth_scale']
        label=Path(root)/s['stream']['relative_dir']/f"labels_{int(s['frames'][1]):06d}.npz"
        with np.load(label) as z:mask=(z['seg']==s['stream']['object_id'])&(observed>0)&(d>0)
        mask=cv2.erode(mask.astype('uint8'),np.ones((5,5),'uint8')).astype(bool)
        residual=(observed-d)[mask]*1000
        rows.append(dict(stream_id=s['stream']['stream_id'],object_id=s['stream']['object_id'],frame=int(s['frames'][1]),pixels=int(mask.sum()),
            median_signed_mm=float(np.median(residual)) if len(residual) else None,median_abs_mm=float(np.median(np.abs(residual))) if len(residual) else None))
    Path(out).write_text(json.dumps(dict(completed=True,rows=rows,depth_correction=False,
        scope='Train-only GT render diagnostic, no input correction. Existing depth/GT mismatch is retained; no fine calibration claim.'),indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--init-from',required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--index-root',default='cache/dexycb_s0');p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=('all','fragments','geometry'),default='all');p.add_argument('--manifest');a=p.parse_args()
    c=load_stream_config(a.config)
    if a.mode=='fragments':fragment_metrics(c,a.init_from,a.data_root,a.index_root,json.loads(Path(a.manifest).read_text()),a.out);return
    if a.mode=='geometry':geometry_check(a.data_root,a.index_root,json.loads(Path(a.manifest).read_text()),a.out);return
    project=Path(__file__).resolve().parents[1];os.chdir(project)
    data=Path(a.data_root).resolve();index=Path(a.index_root).resolve();init=Path(a.init_from).resolve()
    base=Path(a.out).resolve();out=base/{'stream_single':'single','stream_dual':'dual','stream_dual_cross':'dual_cross','stream_dual_cross_residual':'dual_cross_residual','stream_rk_factorial':'rk','stream_rk_spatial':'rk_spatial','stream_rk_aligned':'rk_aligned','stream_rk_direct_pose':'rk_direct_pose','stream_rk_pose_reference':'rk_pose_reference'}[c['architecture_id']];out.mkdir(parents=True,exist_ok=True)
    audit=check_data_gate(index);python=sys.executable
    report=dict(architecture_id=c['architecture_id'],config_hash=config_hash(c),source_sha256=source_hash(),cache_contract=cache_contract_for(c['architecture_id']),
        split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],init_sha256=sha(init),approved=False,checks={})
    def record():
        (out/'approval.json').write_text(json.dumps(report,indent=2))
    def run(command,log,env=None):
        path=out/log
        if path.exists():path=path.with_name(path.stem+'_'+str(time.time_ns())+path.suffix)
        started=time.time()
        with path.open('w') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
        with (out/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,log=str(path),returncode=result.returncode,seconds=time.time()-started))+'\n')
        if result.returncode:raise RuntimeError('Preflight command failed: '+str(path))
    def free_required(stage):
        occupied=occupied_gpus()
        if occupied:
            report['checks'][stage]=dict(status='not_run',reason='GPU compute processes already present',processes=occupied);record()
            return False
        return True
    cpuenv=os.environ.copy();cpuenv.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2')
    try:
        run([python,'-m','pytest','tests','-m','not cuda and not foundationpose','-q','-s','--junitxml='+str(out/'cpu_tests.xml')],'cpu_tests.log',cpuenv)
        tree=ET.parse(out/'cpu_tests.xml');report['checks']['cpu_tests']=dict(status='passed',tests=sum(int(x.get('tests',0)) for x in tree.findall('.//testsuite')));record()
        if not free_required('gpu_preflight'):return
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(project/'src'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
        if not (out/'fixed16.json').exists():
            ds=StreamClips(data,index,c['burn_in_frames'],c['supervised_unroll_frames']);manifest=ds.fixed_manifest(16)
            (out/'fixed16.json').write_text(json.dumps(manifest,indent=2));del ds
        manifest=out/'fixed16.json'
        probe=dict(c,world_size=1,effective_sequences_per_step=c['batch_sequences_per_gpu']*c['grad_accum_steps'],
                   nominal_supervised_updates_per_step=c['batch_sequences_per_gpu']*c['grad_accum_steps']*c['supervised_unroll_frames'])
        overfit=dict(probe,batch_sequences_per_gpu=1,grad_accum_steps=1,effective_sequences_per_step=1,nominal_supervised_updates_per_step=c['supervised_unroll_frames'],save_every=25)
        for name,cfg in [('memory_probe',probe),('overfit',overfit)]:
            cfgpath=out/f'{name}.yaml'
            if cfgpath.exists() and yaml.safe_load(cfgpath.read_text())!=cfg:raise ValueError('Existing preflight config differs: '+str(cfgpath))
            cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False))
        common=['--data-root',str(data),'--index-root',str(index)]
        for name in ['before','after']:
            path=out/f'overfit_{name}.json'
            if name=='before' and not path.exists():
                run([python,str(__file__),'--mode','fragments','--config',str(out/'overfit.yaml'),'--init-from',str(init),*common,'--manifest',str(manifest),'--out',str(path)],'overfit_before.log',env)
        def train_check(name,cfg,steps,world=1,resume_path=None):
            folder=out/name;done=folder/'completed.json';target=steps
            if done.exists() and json.loads(done.read_text())['new_stage_step']>=target:
                saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
                if saved['config_hash']!=config_hash(load_stream_config(cfg)):raise ValueError('Completed stage has different config')
                del saved;return
            if not free_required(name):raise RuntimeError('Resources became occupied; requested stage left unrun')
            cmd=[python]
            stage_env=dict(env)
            if world>1:
                stage_env['CUDA_VISIBLE_DEVICES']=','.join(map(str,range(world)))
                cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node='+str(world)]
            cmd+=['-m','lip.train_stream','--config',str(cfg),'--output',str(folder),*common,'--max-steps',str(steps),'--preflight']
            cmd+=['--resume',str(resume_path)] if resume_path else ['--init-from',str(init)]
            if world==1:cmd+=['--fixed-manifest',str(manifest)]
            run(cmd,name+('_resume.log' if resume_path else '.log'),stage_env)
        train_check('memory_probe',out/'memory_probe.yaml',3)
        report['checks']['full_unroll_memory']=dict(status='passed',records=str(out/'memory_probe/rank0.jsonl'))
        train_check('overfit',out/'overfit.yaml',300)
        if not (out/'overfit_after.json').exists():
            run([python,str(__file__),'--mode','fragments','--config',str(out/'overfit.yaml'),'--init-from',str(out/'overfit/last.pt'),*common,'--manifest',str(manifest),'--out',str(out/'overfit_after.json')],'overfit_after.log',env)
        before=json.loads((out/'overfit_before.json').read_text());after=json.loads((out/'overfit_after.json').read_text())
        report['checks']['overfit']=dict(status='completed',steps=300,real_fragments=16,before_loss=before['mean_loss'],after_loss=after['mean_loss'],
            improved=after['mean_loss']<before['mean_loss'],scope='No fabricated mandatory accuracy target')
        run([python,str(__file__),'--mode','geometry','--config',str(out/'overfit.yaml'),'--init-from',str(init),*common,'--manifest',str(manifest),'--out',str(out/'geometry.json')],'geometry.log',env)
        report['checks']['geometry']=dict(status='checked_with_known_depth_mismatch',correction=False)
        if not free_required('cuda_tests'):return
        cuda_tests=['tests/test_stream_cuda.py']
        if c['architecture_id'] in ('stream_dual_cross','stream_dual_cross_residual'):cuda_tests+=['tests/test_stream_cross_cuda.py']
        if c['architecture_id']=='stream_dual_cross_residual':cuda_tests+=['tests/test_stream_residual_cross.py']
        if c['architecture_id'].startswith('stream_rk'):cuda_tests+=['tests/test_stream_rk_cuda.py']
        if c['architecture_id']=='stream_rk_aligned':cuda_tests+=['tests/test_aligned_memory.py','tests/test_stream_aligned_memory.py']
        if c['architecture_id']=='stream_rk_direct_pose':cuda_tests+=['tests/test_direct_pose_residual.py','tests/test_stream_direct_pose.py']
        if c['architecture_id']=='stream_rk_pose_reference':cuda_tests+=['tests/test_pose_reference.py','tests/test_stream_pose_reference.py']
        run([python,'-m','pytest',*cuda_tests,'-q','-s','--junitxml='+str(out/'cuda_tests.xml')],'cuda_tests.log',env)
        report['checks']['cuda_tests']=dict(status='passed',reference_and_10000_steps=True);record()
        if torch.cuda.device_count()<8:
            report['checks']['ddp']=dict(status='not_run',reason='Fewer than 8 visible CUDA GPUs');record();return
        cfg=Path(a.config).resolve()
        train_check('ddp',cfg,50,8)
        train_check('ddp',cfg,53,8,out/'ddp/last.pt')
        checks=[json.loads((out/f'ddp/resume_rank{r}.json').read_text()) for r in range(8)]
        assert all(r['passed'] and r['loaded_step']==50 and r['rng_restored'] for r in checks)
        report['checks']['ddp']=dict(status='passed',steps=50,resume_steps=3,all_ranks_restored=True)
        report['approved']=True;report['completed']=True;record()
    except Exception as exc:
        report['error']=repr(exc);record();raise

if __name__=='__main__':main()
