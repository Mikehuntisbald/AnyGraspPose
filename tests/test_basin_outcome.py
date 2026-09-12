import pytest,torch
from lip.models.basin import BasinOutcomeCritic,basin_loss,load_frozen_basin

def test_dual_critic_actor_gradient_and_frozen_parameters():
 torch.manual_seed(731);q=BasinOutcomeCritic().eval().requires_grad_(False)
 z=torch.randn(3,256,requires_grad=True);d=torch.randn(3,6,requires_grad=True)
 o=q.outcomes(z.detach(),d);assert set(o)=={'converge_logit','improve_logit','log_after'}
 basin_loss(q,z,d).backward();assert z.grad is None and d.grad is not None and torch.isfinite(d.grad).all() and d.grad.abs().sum()>0
 assert all(p.grad is None for p in q.parameters())

def test_failed_critic_requires_explicit_experimental_policy(tmp_path):
 q=BasinOutcomeCritic();p=tmp_path/'critic.pt';torch.save(dict(architecture='outcome_v2',model=q.state_dict(),receipt={'passed':False}),p)
 with pytest.raises(ValueError,match='quality gate'):load_frozen_basin(p,'cpu')
 loaded=load_frozen_basin(p,'cpu',allow_experimental=True)
 assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
