"""Explicit horizon extension, preserving model/Adam/RNG/sampler at the boundary."""
from copy import deepcopy
import math
import torch
from lip.engine.jepa_checkpoint import sha,load_core,core_state,restore_rng,rng_state
from lip.jepa.config import config_hash
from .checkpoint import software_environment


def extension_factor(step,source_step,rewarm_steps,total_steps,floor=.1):
    if not (0 < rewarm_steps < total_steps-source_step and 0 < floor <= 1):
        raise ValueError('Invalid extension schedule')
    if step <= source_step:return floor
    if step < source_step+rewarm_steps:
        return floor+(1-floor)*(step-source_step)/rewarm_steps
    progress=min(1.,(step-source_step-rewarm_steps)/(total_steps-source_step-rewarm_steps))
    return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))


def scheduler_origin(config):
    """Map a saved scheduler clock to global optimizer steps."""
    return config["migration"]["source_step"] if config["runtime"].get("phase_relative_schedule",False) else 0


def validate_extension_config(old,new):
    """Only the declared horizon, output location and evaluation budget may change."""
    candidate=deepcopy(new);candidate.pop('horizon_continuation')
    candidate['paths']['output']=old['paths']['output']
    candidate['training']['max_steps']=old['training']['max_steps']
    for key in ('steps','history_off_steps','stop_after_step'):
        candidate['validation'][key]=deepcopy(old['validation'][key])
    candidate['budget']=deepcopy(old['budget'])
    if candidate!=old:raise ValueError('Horizon extension changed model, inputs, loss, data or optimizer settings')
    if new['training']['max_steps']<=old['training']['max_steps']:raise ValueError('Non-increasing horizon')


def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def extend_horizon(path,model,optimizer,scheduler,config,provenance,rank,world):
    plan=config['horizon_continuation']
    if sha(path)!=plan['source_sha256']:raise ValueError('Extension checkpoint hash mismatch')
    source=torch.load(path,map_location='cpu',weights_only=False)
    if source['config_hash']!=config_hash(source['config']) or source['config_hash']!=plan['source_config_hash']:
        raise ValueError('Source configuration hash mismatch')
    validate_extension_config(source['config'],config)
    if source['architecture_id']!=model.architecture_id or source['software_environment']!=software_environment():
        raise ValueError('Source model/environment mismatch')
    for key in ['split_hash','mesh_hash','initializers_sha256','weights']:
        if source['provenance'][key]!=provenance[key]:raise ValueError('Extension provenance mismatch: '+key)
    step=plan['source_step']
    if source['step']!=step or len(source['rng'])!=world or source['sampler_position']!=step*config['training']['effective_batch']:
        raise ValueError('Extension sampler/world/step mismatch')
    groups=optimizer.state_dict()['param_groups']
    if [g['names'] for g in groups]!=[g['names'] for g in source['optimizer']['param_groups']]:
        raise ValueError('Parameter order changed')
    load_core(model,source['model'])
    optimizer.load_state_dict(source['optimizer']);scheduler.load_state_dict(source['scheduler'])
    origin=scheduler_origin(source['config'])
    if scheduler.last_epoch!=step-origin:raise ValueError('Unexpected source scheduler position')
    for group,base in zip(optimizer.param_groups,scheduler.base_lrs):
        expected=base*extension_factor(step,step,plan['rewarm_steps'],config['training']['max_steps'],plan['floor'])
        if not math.isclose(group['lr'],expected,rel_tol=1e-10):raise ValueError('Boundary LR is not continuous')
    if not exact(core_state(model),source['model']) or not exact(optimizer.state_dict(),source['optimizer']) or not exact(scheduler.state_dict(),source['scheduler']):
        raise ValueError('Extension changed restored model/optimizer/scheduler tensors')
    restore_rng(source['rng'][rank])
    import numpy as np
    current=rng_state();saved=source['rng'][rank]
    if current['python']!=saved['python'] or not np.array_equal(current['numpy'][1],saved['numpy'][1]) or current['numpy'][2:]!=saved['numpy'][2:]:
        raise ValueError('Host RNG restoration mismatch')
    if not exact(current['torch'],saved['torch']) or not exact(current['cuda'],saved['cuda']):raise ValueError('Device RNG restoration mismatch')
    return step,dict(kind='horizon_extension',source_sha256=plan['source_sha256'],model_exact=True,
        optimizer_exact=True,scheduler_state_exact=True,sampler_position=source['sampler_position'],rng_exact=True,
        all_rank_rng=world,optimizer_states=len(source['optimizer']['state']),boundary_lr=[g['lr'] for g in optimizer.param_groups],
        schedule='100-step continuous rewarm then cosine to new total; lambda explicitly changed',optimizer_reset=False)
