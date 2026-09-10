import os
import random
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'].cpu())


def save(path,model,optimizer,scheduler,step,config,audit,sampler_position,best=None):
    state=rng_state();states=[state]
    if dist.is_initialized():
        states=[None]*dist.get_world_size();dist.all_gather_object(states,state)
        if dist.get_rank()!=0:return
    m=model.module if hasattr(model,'module') else model
    obj=dict(model=m.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),global_step=step,
             config=config,mesh_hash=audit['mesh_hash'],split_hash=audit['split_hash'],rng=states,
             sampler_position=sampler_position,best=best)
    source=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for file in sorted(source.rglob('*.py')):
        digest.update(str(file.relative_to(source)).encode());digest.update(file.read_bytes())
    obj['code_sha256']=digest.hexdigest()
    obj['rgb_weight_source']=config.get('rgb_weights','ResNet50_Weights.IMAGENET1K_V2' if config.get('pretrained') else 'none')
    tmp=str(path)+'.tmp';torch.save(obj,tmp);os.replace(tmp,path)


def resume(path,model,optimizer,scheduler,audit,rank=0):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    for key in ['mesh_hash','split_hash']:
        if obj[key]!=audit[key]:raise ValueError('Checkpoint '+key+' mismatch')
    model.load_state_dict(obj['model']);optimizer.load_state_dict(obj['optimizer']);scheduler.load_state_dict(obj['scheduler'])
    if rank>=len(obj['rng']):raise ValueError('Resume world size differs from checkpoint')
    restore_rng(obj['rng'][rank]);return obj


def record_best(last_path,best_path,score,metrics_path):
    """Keep selection state in last as well as best, so resume cannot forget it."""
    obj=torch.load(last_path,map_location='cpu',weights_only=False)
    obj['best']=score;obj['best_validation_metrics']=str(metrics_path)
    for path in (last_path,best_path):
        tmp=str(path)+'.tmp';torch.save(obj,tmp);os.replace(tmp,path)
