"""Explicit same-tensor checkpoint adaptation for corrected memory semantics."""
import torch
from lip.engine.jepa_checkpoint import sha,load_core,core_state,restore_rng
from lip.jepa.config import config_hash
from .checkpoint import software_environment
from .horizon_resume import exact


def adapt_history(path,model,optimizer,scheduler,config,rank,world):
    plan=config['history_repair']
    if sha(path)!=plan['source_sha256']:raise ValueError('History source identity mismatch')
    source=torch.load(path,map_location='cpu',weights_only=False)
    if (source['architecture_id'],model.architecture_id) not in (('stream_conv_cross_jepa_v10','stream_conv_cross_supported_history_jepa_v10'),('stream_conv_cross_supported_history_jepa_v10','stream_conv_cross_dense_history_jepa_v10')):
        raise ValueError('Unexpected history migration architectures')
    if source['config_hash']!=config_hash(source['config']) or source['software_environment']!=software_environment():
        raise ValueError('History source config/environment mismatch')
    start=config['migration']['source_step']
    if source['step']!=start or len(source['rng'])!=world or source['sampler_position']!=start*config['training']['effective_batch']:
        raise ValueError('History source sampler/world mismatch')
    current=optimizer.state_dict();old=source['optimizer']
    if [g['names'] for g in current['param_groups']]!=[g['names'] for g in old['param_groups']]:
        raise ValueError('History repair changed parameter identity/order')
    load_core(model,source['model']);optimizer.load_state_dict(old)
    if not exact(core_state(model),source['model']) or not exact(optimizer.state_dict(),old):
        raise ValueError('History adaptation changed weights or Adam states')
    # Explicitly declared short repair schedule; keep native sampling and RNG.
    for group,base,lr in zip(optimizer.param_groups,scheduler.base_lrs,scheduler.get_last_lr()):
        group['initial_lr']=base;group['lr']=lr
    restore_rng(source['rng'][rank])
    return start,dict(kind='history_support_and_short_pair',source_sha256=plan['source_sha256'],
        model_exact=True,adam_moments_exact=True,all_rank_rng=world,sampler_position=source['sampler_position'],
        scheduler='explicit bounded repair schedule',memory='empty; new support contract',
        diagnostic_weights_loaded=False)
