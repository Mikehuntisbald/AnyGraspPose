"""Explicit model/optimizer migration for frozen, geometry-grounded readout training."""
import torch
from .reconstruction_only import is_pose_parameter


def install_readout_and_restore_nonpose(model, optimizer, parent, readout):
    if any(p.requires_grad for name,p in model.named_parameters() if is_pose_parameter(name)):
        raise ValueError('Readout must already be frozen')
    current=model.state_dict()
    pose_keys={k for k in current if is_pose_parameter(k)}
    if set(readout)!=pose_keys:
        raise ValueError('Readout parameter coverage mismatch')
    shared={k:v for k,v in parent['model'].items() if not is_pose_parameter(k)}
    if set(shared)!={k for k in current if not is_pose_parameter(k)}:
        raise ValueError('Shared restoration architecture changed')
    with torch.no_grad():
        for k,v in {**shared,**readout}.items():
            if v.shape!=current[k].shape:raise ValueError('Tensor shape mismatch: '+k)
            current[k].copy_(v)
            if not torch.equal(current[k].cpu(),v.cpu()):raise ValueError('Inexact migration: '+k)
    old=parent['optimizer'];old_by_name={name:pid for group in old['param_groups'] for name,pid in zip(group['names'],group['params'])}
    fresh=optimizer.state_dict();copied=0;without_state=[]
    for group in fresh['param_groups']:
        for name,pid in zip(group['names'],group['params']):
            if is_pose_parameter(name):raise ValueError('Frozen readout entered optimizer')
            if name not in old_by_name:raise ValueError('Unexpected new restoration parameter: '+name)
            old_id=old_by_name[name]
            if old_id in old['state']:
                fresh['state'][pid]=old['state'][old_id];copied+=1
            else:without_state.append(name)
    optimizer.load_state_dict(fresh)
    restored=optimizer.state_dict()
    for group in restored['param_groups']:
        for name,pid in zip(group['names'],group['params']):
            if old_by_name[name] not in old['state']:continue
            for k,v in old['state'][old_by_name[name]].items():
                other=restored['state'][pid][k]
                if isinstance(v,torch.Tensor):
                    if not torch.equal(v.cpu(),other.cpu()):raise ValueError('Inexact Adam state: '+name)
                elif v!=other:raise ValueError('Inexact optimizer metadata: '+name)
    return dict(shared_model_tensors=len(shared),readout_tensors=len(readout),restored_optimizer_states=copied,
                parameters_without_existing_adam_state=without_state,restoration_model_and_adam_exact=True,readout_frozen=True)
