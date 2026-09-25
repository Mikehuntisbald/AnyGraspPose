from copy import deepcopy
import torch
from lip.unified.joint_pose import inherit_optimizer,enable_joint_pose
from lip.unified.horizon_resume import exact


def test_inherit_moments_by_name_when_groups_expand():
    a=torch.nn.Parameter(torch.tensor([2.]));b=torch.nn.Parameter(torch.tensor([3.]))
    old=torch.optim.AdamW([dict(params=[a,b],names=['core.a','surface_head.b'])],lr=1e-5,betas=(.9,.95))
    (a.square()+b.square()).backward();old.step();source={'optimizer':deepcopy(old.state_dict())}
    head=torch.nn.Parameter(torch.tensor([4.]))
    new=torch.optim.AdamW([dict(params=[b],names=['surface_head.b']),dict(params=[head],names=['head.weight']),dict(params=[a],names=['core.a'])],lr=3e-6,betas=(.9,.95))
    report=inherit_optimizer(new,source)
    assert report['inherited_tensors']==2 and report['new_readout_tensors']==1
    assert exact(new.state[b],old.state[b]) and exact(new.state[a],old.state[a])
    assert not new.state[head]


def test_unfreeze_full_readout_only():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.query=torch.nn.Parameter(torch.zeros(2));self.head=torch.nn.Linear(2,2);self.object_attn=torch.nn.Linear(2,2);self.geometry_readout=torch.nn.Linear(2,2);self.encoder=torch.nn.Linear(2,2)
    m=Model().requires_grad_(False);enable_joint_pose(m)
    assert all(p.requires_grad for n,p in m.named_parameters() if not n.startswith('encoder.'))
    assert all(not p.requires_grad for p in m.encoder.parameters())
