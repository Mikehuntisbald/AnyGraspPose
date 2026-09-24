"""Explicit LR-only intervention; inherit tensors, Adam moments and global schedule clock."""
from copy import deepcopy
from .horizon_resume import extension_factor, exact


def rate_factor(config, offset):
    p=config['lr_intervention']
    if not 0 < p['scale'] <= 1:raise ValueError('LR scale must be in (0,1]')
    return p['scale']*extension_factor(config['migration']['source_step']+offset,
        p['schedule_origin'],p['rewarm_steps'],p['schedule_end'],p['floor'])


def apply_rates(optimizer, scheduler, source, config):
    before=deepcopy(optimizer.state_dict())
    if not exact(before,source['optimizer']):raise ValueError('Adam source mismatch before LR intervention')
    old=[g['lr'] for g in optimizer.param_groups]
    for group,lr in zip(optimizer.param_groups,scheduler.get_last_lr()):group['lr']=lr
    after=deepcopy(optimizer.state_dict())
    for a,b in zip(after['param_groups'],before['param_groups']):a['lr']=b['lr']
    if not exact(after,before):raise ValueError('LR intervention changed non-LR optimizer state')
    return dict(old=old,new=scheduler.get_last_lr(),non_lr_adam_exact=True,plan=config['lr_intervention'])
