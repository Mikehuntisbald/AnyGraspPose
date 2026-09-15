import torch
from lip.models.update_quality import UpdateQualityHead,observation_vector
from lip.geometry.so3 import update


def test_frozen_quality_head_still_differentiates_candidate_action():
    torch.manual_seed(42);model=UpdateQualityHead().eval()
    for p in model.parameters():p.requires_grad_(False)
    observation=torch.randn(1,815).expand(2,-1);delta=torch.tensor([[0.,0.,0.,0.,0.,0.],[.1,0.,0.,.02,0.,0.]],requires_grad=True)
    out=model(observation,delta)
    assert not torch.equal(out['harm_logit'][0],out['harm_logit'][1])
    out['harm_logit'].sum().backward();assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters())


def test_observation_contract_has_no_target_or_hand_fields():
    b=2;g=torch.zeros(b,9,16,16);g[:,3]=1;g[:,8]=1
    f=dict(geometry=g,state_input=torch.randn(b,24))
    out={k:torch.randn(b,256) for k in ('latent','latent_object','latent_context')}
    out.update(delta_rotvec=torch.randn(b,3),delta_center_norm=torch.randn(b,3))
    vector,delta=observation_vector(f,out)
    assert vector.shape==(b,815) and delta.shape==(b,6) and torch.isfinite(vector).all()
    f['unused_gt']=torch.full((b,4,4),float('nan'));f['unused_hand']=None
    actual,_=observation_vector(f,out);assert torch.equal(vector,actual)


def test_zero_candidate_exactly_preserves_base_pose():
    base=torch.eye(4).repeat(2,1,1);base[:,2,3]=.6
    pose=update(base,torch.zeros(2,3),torch.zeros(2,3),torch.tensor([.1,.2]))
    assert torch.equal(pose,base)
