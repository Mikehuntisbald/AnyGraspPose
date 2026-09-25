"""Joint recovery/pose continuation with name-keyed Adam migration."""
from copy import deepcopy
import torch
from .reconstruction_only import recovery_objective,is_pose_parameter
from .horizon_resume import exact


def enable_joint_pose(model):
    for name,parameter in model.named_parameters():
        if is_pose_parameter(name):parameter.requires_grad_(True)


def joint_objective(output,teacher,truth,points,diameter,weights):
    from lip.losses import pose_loss
    recovery,metrics=recovery_objective(output,teacher,truth,points,diameter,weights)
    pose,parts=pose_loss(output['pose_centered'],truth,points,diameter)
    pose_metrics=torch.stack([pose.detach(),parts['translation'].detach(),parts['rotation'].detach(),parts['points'].detach()])
    return recovery+weights.get('pose',1.)*pose,torch.cat((pose_metrics,metrics[4:]))


def inherit_optimizer(optimizer,source):
    """Old parameter moments remain exact; only newly trainable readout starts empty."""
    old=source['optimizer'];new=optimizer.state_dict();old_params={}
    for group in old['param_groups']:
        for name,index in zip(group['names'],group['params']):old_params[name]=(index,group)
    inherited=[];fresh=[]
    for group in new['param_groups']:
        for name,index in zip(group['names'],group['params']):
            if name in old_params:
                old_index,old_group=old_params[name]
                for key in ('betas','eps','weight_decay','amsgrad'):
                    if group[key]!=old_group[key]:raise ValueError('Optimizer hyperparameter changed: '+name+'/'+key)
                if old_index in old['state']:new['state'][index]=deepcopy(old['state'][old_index])
                inherited.append(name)
            else:
                if not is_pose_parameter(name):raise ValueError('Unexpected newly trainable tensor: '+name)
                fresh.append(name)
    if set(inherited)!=set(old_params):raise ValueError('Continuation dropped existing trainable tensors')
    if not fresh:raise ValueError('No pose readout parameters were unfrozen')
    optimizer.load_state_dict(new)
    actual=optimizer.state_dict()
    for group in actual['param_groups']:
        for name,index in zip(group['names'],group['params']):
            if name in old_params:
                old_index,_=old_params[name]
                if not exact(actual['state'].get(index,{}),old['state'].get(old_index,{})):raise ValueError('Adam moments changed')
            elif index in actual['state']:raise ValueError('New readout Adam was not empty')
    return dict(inherited_tensors=len(inherited),new_readout_tensors=len(fresh),inherited_moments_exact=True,new_readout_state_empty=True)
