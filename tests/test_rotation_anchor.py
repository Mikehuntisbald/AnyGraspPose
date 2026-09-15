import torch
import pytest
from lip.geometry.so3 import exp
from lip.models.rotation_anchor import RotationAnchorReadout,angular_evidence,blend_rotation_reference


def test_geodesic_rotation_read_preserves_center_zero_identity_and_near_pi_endpoint():
    reference=torch.eye(4).repeat(2,1,1);reference[:,:3,3]=torch.tensor([[.2,-.1,.7],[.1,.1,.9]])
    reference[:,:3,:3]=exp(torch.tensor([[.1,-.2,.3],[0.,0.,.05]]))
    anchor=exp(torch.tensor([[.3,.1,-.2],[0.,0.,torch.pi-.0001]]))
    assert torch.equal(blend_rotation_reference(reference,anchor,torch.zeros(2)),reference)
    full=blend_rotation_reference(reference,anchor,torch.ones(2))
    torch.testing.assert_close(full[:,:3,:3],anchor,rtol=1e-5,atol=1e-5)
    for fraction in (0.,.2,.5,1.):
        value=blend_rotation_reference(reference,anchor,torch.full((2,),fraction))
        assert torch.equal(value[:,:3,3],reference[:,:3,3])
        rotation=value[:,:3,:3]
        torch.testing.assert_close(rotation.transpose(-1,-2)@rotation,torch.eye(3).expand(2,-1,-1),atol=1e-5,rtol=1e-5)
        assert (torch.linalg.det(rotation)>0).all()


def test_zero_anchor_head_can_learn_and_measurements_do_not_receive_gradients():
    head=RotationAnchorReadout();latent=torch.randn(2,256);state=torch.randn(2,24)
    reference=torch.eye(4).repeat(2,1,1);reference[:,2,3]=.7;reference.requires_grad_()
    anchor=exp(torch.tensor([[.2,.1,0.],[.1,.2,0.]])).requires_grad_()
    visual=reference.detach().clone().requires_grad_()
    evidence=angular_evidence(reference,anchor,visual)
    fraction=head(latent,state,evidence,torch.ones(2),torch.zeros(2))
    assert not fraction.count_nonzero()
    mixed=blend_rotation_reference(reference,anchor,fraction)
    mixed[:,0,2].sum().backward()
    assert head.readout[-1].weight.grad.abs().sum()>0
    assert anchor.grad is None and visual.grad is None and torch.isfinite(reference.grad).all()
    head.readout[-1].bias.data.fill_(2.)
    assert torch.equal(head(latent,state,evidence.detach(),torch.ones(2),torch.ones(2)),torch.ones(2))
    with pytest.raises(ValueError):head(latent,state[:,:2],evidence,torch.ones(2),torch.ones(2))
