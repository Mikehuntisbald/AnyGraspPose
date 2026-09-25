import torch
from torch import nn
from lip.unified.readout_adaptation import install_readout_and_restore_nonpose


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.restoration=nn.Linear(2,3);self.head=nn.Linear(3,1)


def test_named_adam_migration_preserves_restoration_and_replaces_frozen_head():
    old=Toy();pairs=list(old.named_parameters())
    optimizer=torch.optim.AdamW([dict(params=[p for n,p in pairs],names=[n for n,p in pairs])],lr=.01)
    old.head(old.restoration(torch.ones(2,2))).sum().backward();optimizer.step()
    parent=dict(model=old.state_dict(),optimizer=optimizer.state_dict())
    new=Toy();new.head.requires_grad_(False)
    # Different parameter ordering and no pose group must still restore by name.
    pairs=list(reversed([(n,p) for n,p in new.named_parameters() if p.requires_grad]))
    opt=torch.optim.AdamW([dict(params=[p for n,p in pairs],names=[n for n,p in pairs])],lr=.001)
    readout={k:v+1 for k,v in parent['model'].items() if k.startswith('head.')}
    receipt=install_readout_and_restore_nonpose(new,opt,parent,readout)
    assert receipt['restored_optimizer_states']==2
    for k,v in new.state_dict().items():torch.testing.assert_close(v,readout[k] if k in readout else parent['model'][k])
    assert opt.param_groups[0]['lr']==.001
    for name,parameter in pairs:
        old_id=parent['optimizer']['param_groups'][0]['names'].index(name)
        for key,value in opt.state[parameter].items():torch.testing.assert_close(value,parent['optimizer']['state'][old_id][key])
