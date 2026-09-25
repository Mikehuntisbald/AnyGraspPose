"""Train pose on the unified JEPA patch AND its decoded content, no encoder skip."""
import torch
from torch import nn
from .serial_completion import CompletionRelations


class SharedPatchRelations(CompletionRelations):
    def __init__(self, patch_scale=1.):
        super().__init__()
        if patch_scale not in (0.,1.):raise ValueError('Paired probe requires patch scale0 or1')
        self.patch_scale=float(patch_scale)
        self.patch_projection=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128))
        nn.init.zeros_(self.patch_projection[-1].weight)
        nn.init.zeros_(self.patch_projection[-1].bias)

    def encode_appearance(self, packet):
        # No observation/renderer/FP representation can enter here. The only
        # extra input is the same patch that feeds the DINO and DPT decoders.
        return super().encode_appearance(packet)+self.patch_scale*self.patch_projection(packet['patch_feature'])


def restore_with_patch_extension(model,optimizer,parent):
    """Preserve every existing model and named Adam tensor; only adapter is new."""
    prefix='geometry_readout.patch_projection.'
    current=model.state_dict();old=parent['model']
    added=set(current)-set(old)
    if not added or set(old)-set(current) or any(not k.startswith(prefix) for k in added):
        raise ValueError('Unexpected shared-patch architecture migration')
    with torch.no_grad():
        for name,value in old.items():
            if current[name].shape!=value.shape:raise ValueError('Changed shape: '+name)
            current[name].copy_(value)
            if not torch.equal(current[name].cpu(),value.cpu()):raise ValueError('Inexact model restore: '+name)
    oldopt=parent['optimizer'];by_name={n:i for g in oldopt['param_groups'] for n,i in zip(g['names'],g['params'])}
    fresh=optimizer.state_dict();copied=0;new=[]
    for group in fresh['param_groups']:
        for name,pid in zip(group['names'],group['params']):
            if name not in by_name:
                if not name.startswith(prefix):raise ValueError('Unexpected optimizer parameter: '+name)
                new.append(name)
            elif by_name[name] in oldopt['state']:
                fresh['state'][pid]=oldopt['state'][by_name[name]];copied+=1
    optimizer.load_state_dict(fresh)
    restored=optimizer.state_dict()
    for group in restored['param_groups']:
        for name,pid in zip(group['names'],group['params']):
            if name not in by_name or by_name[name] not in oldopt['state']:continue
            for key,value in oldopt['state'][by_name[name]].items():
                other=restored['state'][pid][key]
                if isinstance(value,torch.Tensor):
                    if not torch.equal(value.cpu(),other.cpu()):raise ValueError('Inexact Adam restore: '+name)
                elif value!=other:raise ValueError('Inexact Adam metadata: '+name)
    return dict(existing_model_tensors_exact=len(old),existing_adam_states_exact=copied,
        added_model_keys=sorted(added),new_optimizer_parameters=new,pose_head_reset=False)
