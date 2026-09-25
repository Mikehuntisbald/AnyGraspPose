import torch
from lip.unified.shape_conditioned import normalized_relation
from lip.unified.serial_completion import CompletionRelations
from lip.geometry.so3 import exp


def test_rotation_signal_sign_and_shape_scale():
    theta=torch.tensor([[.03,-.04,.02]])
    r=exp(theta)
    for radii in ([.1,.2,.3],[.01,.2,.3],[1.,2.,3.]):
        scatter=torch.diag(torch.tensor(radii).square())[None]
        covariance=scatter@r.transpose(1,2)
        zeros=torch.zeros(1,3)
        signal=normalized_relation(scatter,covariance,zeros,zeros)
        torch.testing.assert_close(signal[:,:3]*.1745329252,theta,rtol=.1,atol=.003)
        assert signal[:,3:].abs().max()==0


def test_degenerate_surface_finite_and_differentiable():
    scatter=torch.diag(torch.tensor([.1,0.,0.]))[None].requires_grad_()
    cov=scatter.clone();p=torch.tensor([[.01,.02,.03]],requires_grad=True)
    result=normalized_relation(scatter,cov,p,p+.01)
    assert result.isfinite().all()
    result.sum().backward()
    assert scatter.grad.isfinite().all() and p.grad.isfinite().all()
    zeros=torch.zeros_like(scatter)
    assert normalized_relation(zeros,zeros,torch.zeros_like(p),torch.zeros_like(p)).abs().max()==0


def test_legacy_parameter_identity_and_zero_new_projection():
    torch.manual_seed(42);old=CompletionRelations()
    torch.manual_seed(42);new=CompletionRelations('shape_conditioned')
    for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k])
    assert new.precondition[-1].weight.count_nonzero()==0
