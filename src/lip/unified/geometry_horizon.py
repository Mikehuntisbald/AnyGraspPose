"""Explicit recovery-only horizon extension; preserve Adam/RNG/sampler state."""
from copy import deepcopy
import math
import numpy as np
import torch
from lip.engine.jepa_checkpoint import sha, core_state, load_core, rng_state, restore_rng
from lip.jepa.config import config_hash
from .checkpoint import software_environment
from .horizon_resume import exact, extension_factor


def validate_geometry_extension(old,new):
    candidate=deepcopy(new)
    plan=candidate.pop('geometry_horizon_continuation')
    candidate['paths']['output']=old['paths']['output']
    candidate['geometry_transport_training']['updates']=old['geometry_transport_training']['updates']
    if candidate!=old:
        raise ValueError('Geometry extension changed inputs, loss, architecture or optimizer settings')
    if new['geometry_transport_training']['updates']<=plan['source_step']:
        raise ValueError('Non-increasing geometry horizon')


def extend_geometry(path,model,optimizer,scheduler,config,provenance,rank,world):
    plan=config['geometry_horizon_continuation']
    if sha(path)!=plan['source_sha256']:raise ValueError('Geometry extension checkpoint hash mismatch')
    source=torch.load(path,map_location='cpu',weights_only=False)
    if source['config_hash']!=config_hash(source['config']) or source['config_hash']!=plan['source_config_hash']:
        raise ValueError('Source config hash mismatch')
    validate_geometry_extension(source['config'],config)
    if source['architecture_id']!=model.architecture_id or source['software_environment']!=software_environment():
        raise ValueError('Source architecture/environment mismatch')
    for key in ('split_hash','mesh_hash','initializers_sha256','frozen_parameter_names'):
        if source['provenance'][key]!=provenance[key]:raise ValueError('Geometry extension provenance mismatch: '+key)
    step=plan['source_step']
    if source['step']!=step or len(source['rng'])!=world or source['sampler_position']!=step*config['training']['effective_batch']:
        raise ValueError('Source world/step/sampler mismatch')
    if [g['names'] for g in optimizer.param_groups]!=[g['names'] for g in source['optimizer']['param_groups']]:
        raise ValueError('Optimizer parameter order changed')
    load_core(model,source['model'])
    optimizer.load_state_dict(source['optimizer']);scheduler.load_state_dict(source['scheduler'])
    if scheduler.last_epoch!=step:raise ValueError('Unexpected geometry scheduler clock')
    for g,base in zip(optimizer.param_groups,scheduler.base_lrs):
        expected=base*extension_factor(step,step,plan['rewarm_steps'],config['geometry_transport_training']['updates'],plan['floor'])
        if not math.isclose(g['lr'],expected,rel_tol=1e-10):raise ValueError('Discontinuous boundary LR')
    if not exact(core_state(model),source['model']) or not exact(optimizer.state_dict(),source['optimizer']) or not exact(scheduler.state_dict(),source['scheduler']):
        raise ValueError('Geometry extension changed restored state')
    restore_rng(source['rng'][rank]);now=rng_state();saved=source['rng'][rank]
    if now['python']!=saved['python'] or not np.array_equal(now['numpy'][1],saved['numpy'][1]) or now['numpy'][2:]!=saved['numpy'][2:]:
        raise ValueError('Host RNG mismatch')
    if not exact(now['torch'],saved['torch']) or not exact(now['cuda'],saved['cuda']):raise ValueError('Device RNG mismatch')
    return step,dict(verified=True,source_sha256=plan['source_sha256'],model_exact=True,optimizer_exact=True,
        scheduler_state_exact=True,rng_exact=True,sampler_position=source['sampler_position'],all_rank_rng=world,
        optimizer_states=len(source['optimizer']['state']),boundary_lr=[g['lr'] for g in optimizer.param_groups],
        schedule='Explicit continuous rewarm then cosine extension; scheduler lambda changes, saved state does not',optimizer_reset=False)
