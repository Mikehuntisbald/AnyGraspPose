"""Explicit objective/config adaptation after offline whole-backbone training."""
import torch
from lip.engine.jepa_checkpoint import sha,load_core,restore_rng
from .checkpoint import software_environment


def adapt_curriculum(path,model,optimizer,scheduler,config,rank,world):
    expected=config['curriculum_continuation']
    if sha(path)!=expected['sha256']:raise ValueError('Curriculum checkpoint identity mismatch')
    source=torch.load(path,map_location='cpu',weights_only=False)
    if source['architecture_id']!=model.architecture_id or source['software_environment']!=software_environment():
        raise ValueError('Curriculum model/environment mismatch')
    if not source.get('curriculum',{}).get('train_only'):raise ValueError('Expected train-only curriculum')
    start=config['migration']['source_step']
    if source['step']!=start or source['sampler_position']!=start*config['training']['effective_batch'] or len(source['rng'])!=world:
        raise ValueError('Native sampler/RNG continuation mismatch')
    if not model.pose_final_patch_only or not model.pose_patch_only or hasattr(model,'pose_relation_proj'):
        raise ValueError('Final-patch-only architecture required')
    load_core(model,source['model'])
    old=source['optimizer'];current=optimizer.state_dict()
    by_name={n:old['state'][i] for g in old['param_groups'] for n,i in zip(g['names'],g['params']) if i in old['state']}
    for g in current['param_groups']:
        for n,i in zip(g['names'],g['params']):
            if n in by_name:current['state'][i]=by_name[n]
    optimizer.load_state_dict(current)
    # Preserve moments, but intentionally begin the newly declared phase LR schedule.
    for g,base,lr in zip(optimizer.param_groups,scheduler.base_lrs,scheduler.get_last_lr()):
        g['initial_lr']=base;g['lr']=lr
    restore_rng(source['rng'][rank])
    return start,dict(preserved_optimizer_states=len(by_name),mapping='parameter names',
        scheduler='explicit new phase',native_sampler_restored=True,all_rank_rng=world)
