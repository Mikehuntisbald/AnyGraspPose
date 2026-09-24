"""Audited training adaptation after a train-only pose-readout calibration."""
import copy
import torch
from lip.engine.jepa_checkpoint import load_core, restore_rng, sha
from .checkpoint import software_environment

CALIBRATED_PREFIXES=('head.','object_attn.','object_norm.','readout.','pose_relation_proj.')


def transfer_optimizer_state(optimizer, saved):
    """Parameter IDs change when the relation projection is added: map by name."""
    by_name={name:saved['state'][key] for group in saved['param_groups']
             for name,key in zip(group['names'],group['params']) if key in saved['state']}
    current=optimizer.state_dict();kept=[];reset=[]
    for group in current['param_groups']:
        for name,key in zip(group['names'],group['params']):
            if name in by_name and not name.startswith(CALIBRATED_PREFIXES):
                current['state'][key]=copy.deepcopy(by_name[name]);kept.append(name)
            else:reset.append(name)
    optimizer.load_state_dict(current)
    return dict(preserved=kept,reset=reset,mapping='exact parameter names')


def adapt(path,model,optimizer,scheduler,config,rank,world,rate):
    repair=config['pose_repair']
    if sha(path)!=repair['candidate_sha256']:raise ValueError('Repair candidate identity mismatch')
    if sha(repair['source_checkpoint'])!=repair['source_sha256']:raise ValueError('Repair source identity mismatch')
    source=torch.load(repair['source_checkpoint'],map_location='cpu',weights_only=False)
    candidate=torch.load(path,map_location='cpu',weights_only=False)
    calibration=candidate.get('pose_calibration',{})
    if not calibration.get('train_only') or calibration.get('source_sha256')!=repair['source_sha256']:
        raise ValueError('Not a train-only calibration of this source')
    if source['architecture_id']!=model.architecture_id or source['software_environment']!=software_environment():
        raise ValueError('Repair source architecture/environment changed')
    start=source['step']
    if len(source['rng'])!=world or source['sampler_position']!=start*config['training']['effective_batch']:
        raise ValueError('Repair source sampler/world mismatch')
    # All non-calibrated weights must remain byte-identical to the source.
    for name,value in source['model'].items():
        if not name.startswith(CALIBRATED_PREFIXES) and not torch.equal(value,candidate['model'][name]):
            raise ValueError('Unexpected calibrated parameter: '+name)
    load_core(model,candidate['model'])
    receipt=transfer_optimizer_state(optimizer,source['optimizer'])
    scheduler.load_state_dict(source['scheduler'])
    # New training LR policy, same global schedule phase and step counter.
    scheduler.base_lrs=[config['training']['learning_rates'][g['category']] for g in optimizer.param_groups]
    scheduler._last_lr=[lr*rate(scheduler.last_epoch) for lr in scheduler.base_lrs]
    for g,base,lr in zip(optimizer.param_groups,scheduler.base_lrs,scheduler._last_lr):
        g['initial_lr']=base;g['lr']=lr
    restore_rng(source['rng'][rank])
    return start,receipt
