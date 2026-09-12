import argparse
from datetime import timedelta
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lip.data.clips import ClipDataset,collate
from lip.engine.config import load_config,environment,optimizer_and_scheduler,check_data_gate
from lip.engine.checkpoint import save,resume,record_best
from lip.engine.runtime import batch_step
from lip.geometry.renderer import Renderer
from lip.models.tracker import Tracker


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data-root',default=os.getenv('DEX_YCB_DIR'))
    p.add_argument('--index-root',default='cache/dexycb_s0');p.add_argument('--output',default='runs/lip_v1_s0')
    p.add_argument('--max-steps',type=int);p.add_argument('--resume');p.add_argument('--force-rollout',type=int,choices=[1,4])
    p.add_argument('--allow-verified-subset',action='store_true')
    a=p.parse_args()
    if not a.data_root:p.error('DEX_YCB_DIR or --data-root is required')
    local=int(os.getenv('LOCAL_RANK','0'));world=int(os.getenv('WORLD_SIZE','1'));rank=int(os.getenv('RANK','0'))
    if torch.cuda.is_available():torch.cuda.set_device(local)
    if a.allow_verified_subset and (not a.max_steps or a.max_steps>100):p.error('Verified subset is only allowed for <=100-step diagnostics')
    c=load_config(a.config);audit=check_data_gate(a.index_root,a.allow_verified_subset)
    c['data_complete']=audit['complete'];c['diagnostic_subset']=a.allow_verified_subset
    validation_group=None
    if world>1:
        dist.init_process_group('nccl')
        # Full validation runs on rank 0 and can exceed NCCL's 10-minute
        # collective timeout. Keep this idle wait on a separate CPU group.
        validation_group=dist.new_group(backend='gloo',timeout=timedelta(hours=6))
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    random.seed(c['seed']+rank);np.random.seed(c['seed']+rank);torch.manual_seed(c['seed']+rank)
    torch.set_num_threads(c.get('cpu_threads',2))
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        env=environment();(output/'environment.json').write_text(json.dumps(env,indent=2))
        print(json.dumps(dict(environment=env)),flush=True)
    model=Tracker(c['pretrained']).to(device);optimizer,scheduler=optimizer_and_scheduler(model,c)
    step=0;position=0;best=-float('inf')
    if a.resume:
        ck=resume(a.resume,model,optimizer,scheduler,audit,rank);step=ck['global_step'];position=ck['sampler_position'];best=ck.get('best',best)
        for key in ['batch_size_per_gpu','grad_accum_steps','clip_length','max_optimizer_steps','warmup_optimizer_steps']:
            if ck['config'][key]!=c[key]:raise ValueError('Resume config mismatch: '+key)
        print(json.dumps(dict(resumed_step=step,scheduler_step=scheduler.last_epoch,sampler_position=position,rank=rank)),flush=True)
    if world>1:model=DDP(model,device_ids=[local],broadcast_buffers=False)
    renderer=Renderer(device);batch=c['batch_size_per_gpu'];accum=c['grad_accum_steps'];limit=a.max_steps or c['max_optimizer_steps']
    ds=ClipDataset(a.data_root,a.index_root,c['clip_length'],seed=c['seed'],steps=max(1,(limit-step)*batch*accum),
                   rank=rank,world=world,batch=batch,start=position,augmentation=c.get('augmentation',True),synthetic_erasure=c.get('synthetic_erasure',False))
    workers=c['num_workers_per_rank']
    loader=DataLoader(ds,batch_size=batch,num_workers=workers,collate_fn=collate,pin_memory=True,
                      **(dict(persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn') if workers else {}))
    iterator=iter(loader);writer=SummaryWriter(str(output/'tensorboard')) if rank==0 else None
    logfile=(output/f'rank{rank}.jsonl').open('a');model.train()
    fp_transition=None
    if c.get('fpaware_enabled'):
        from lip.integrations.frozen_fp import FrozenFoundationPose
        fp_transition=FrozenFoundationPose(c['foundationpose_root'],a.data_root,device,c['foundationpose_refiner_sha256'])
        print(json.dumps(dict(fp_refiner_sha256=fp_transition.weight_sha256,fp_frozen=True,rank=rank)),flush=True)
    basin_critic=None
    if c.get('basin_enabled'):
        from lip.models.basin import load_frozen_basin
        experimental=c.get('basin_quality_policy')=='experimental_user_authorized'
        with torch.random.fork_rng(devices=[local] if device.type=='cuda' else []):
            basin_critic=load_frozen_basin(c['basin_checkpoint'],device,c['basin_checkpoint_sha256'],experimental)
        print(json.dumps(dict(basin_critic_frozen=True,basin_checkpoint_sha256=c['basin_checkpoint_sha256'],basin_quality_policy=c.get('basin_quality_policy','validated'),rank=rank)),flush=True)
    while step<limit:
        optimizer.zero_grad(set_to_none=True);records=[];begin=time.perf_counter()
        rng=np.random.default_rng(c['seed']+step)
        history_mode='lip_only';history_probs=None
        u=a.force_rollout or (4 if step>=c['rollout_start_step'] and rng.random()<.5 else 1)
        if c.get('fpaware_enabled') and step>=c['fpaware_start_step']:
            from lip.engine.rollout import choose
            history_mode,history_probs=choose(c,step);u=4
        basin_weight=c.get('basin_lambda_max',0.)*min(1.,max(0.,(step-c.get('basin_start_step',step))/max(1,c.get('basin_warmup_steps',1000)))) if basin_critic is not None else 0.
        for micro in range(accum):
            t=time.perf_counter();items=next(iterator);data_time=time.perf_counter()-t
            values,_=batch_step(model,items,renderer,c,u,c.get('history_noise',True),True,1/accum,micro==accum-1,history_mode,fp_transition,basin_critic,basin_weight)
            values['data_time']=data_time;records.append(values);position+=1
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip_norm'],error_if_nonfinite=True)
        optimizer.step();scheduler.step();step+=1
        elapsed=time.perf_counter()-begin
        values={k:sum(r[k] for r in records)/accum for k in records[0]}
        values.update(step=step,rank=rank,run_id=c.get("run_id","legacy"),rollout=u,history_mode=history_mode,history_probs=history_probs,basin_weight=basin_weight,grad_norm=float(norm),scheduler_step=scheduler.last_epoch,
                      sampler_position=position,samples_seen=position*batch*world,effective_batch=batch*accum*world,
                      clips_per_sec=batch*accum/elapsed,observed_frames_per_sec=batch*accum*c['clip_length']*u/elapsed,
                      peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                      elapsed=elapsed,nonfinite_count=0,lrs=[g['lr'] for g in optimizer.param_groups])
        logfile.write(json.dumps(values)+'\n');logfile.flush()
        if rank==0:
            print(json.dumps(values),flush=True)
            for k,v in values.items():
                if isinstance(v,(int,float)):writer.add_scalar(k,v,step)
        if step%c['save_every_optimizer_steps']==0 or step%c['full_val_every_optimizer_steps']==0 or step==limit:
            save(output/'last.pt',model,optimizer,scheduler,step,c,audit,position,best)
        full=step%c['full_val_every_optimizer_steps']==0
        quick=step%c['quick_val_every_optimizer_steps']==0
        if (full or quick) and not a.max_steps:
            if world>1:dist.barrier(group=validation_group)
            if rank==0:
                from lip.evaluate import evaluate
                report=evaluate(model.module if hasattr(model,'module') else model,c,a.data_root,a.index_root,
                                output/f'val_{step}',split='val',mode='closed-loop',limit_streams=None if full else 8)
                score=report['macro_object']['adds_01']
                if full and score>best:
                    best=score
                    record_best(output/'last.pt',output/'best.pt',best,output/f'val_{step}'/'metrics.json')
                model.train()
            if world>1:
                v=[best];dist.broadcast_object_list(v,0,group=validation_group);best=v[0]
                dist.barrier(group=validation_group)
    logfile.close()
    if writer:writer.close()
    if world>1:
        dist.destroy_process_group(validation_group)
        dist.destroy_process_group()


if __name__=='__main__':main()
