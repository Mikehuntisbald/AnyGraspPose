import pytest
import torch
from lip.geometry.so3 import exp,log
from lip.models.adaptive_reference import ReferenceWriter,write_innovation,write_reference


def poses():
    reference=torch.eye(4).repeat(2,1,1);reference[:,:3,3]=torch.tensor([[.3,-.2,1.],[.1,.2,.8]])
    measurement=reference.clone();measurement[:,:3,:3]=exp(torch.tensor([[.2,-.3,.1],[-.1,.1,torch.pi-1e-4]]))
    measurement[:,:3,3]+=torch.tensor([[.03,.04,-.02],[-.01,.03,.05]])
    return reference,measurement,torch.tensor([.2,.1])


def test_reference_writer_zero_boundary_has_gradient_and_bounded_fraction():
    head=ReferenceWriter();args=(torch.randn(2,256),torch.randn(2,24),torch.randn(2,6),torch.ones(2),torch.ones(2))
    output=head(*args);assert torch.count_nonzero(output)==0
    output.sum().backward();assert (head.readout[-1].weight.grad.abs().sum(1)>0).all()
    head.readout[-1].bias.data.copy_(torch.tensor([-2.,2.]));out=head(*args)
    assert torch.equal(out,torch.tensor([[0.,.25],[0.,.25]]))
    with pytest.raises(ValueError):head(args[0][:,None],*args[1:])


def test_write_units_zero_exact_and_full_target_including_near_pi():
    reference,measurement,diameter=poses();innovation,valid=write_innovation(reference,measurement,diameter)
    assert valid.all()
    zero,_,_=write_reference(reference,innovation,torch.zeros(2,2),diameter);assert torch.equal(zero,reference)
    full,_,_=write_reference(reference,innovation,torch.ones(2,2),diameter)
    torch.testing.assert_close(full,measurement,atol=1e-6,rtol=1e-6)
    rotation,_,_=write_reference(reference,innovation,torch.tensor([[.25,0.],[.25,0.]]),diameter)
    assert torch.equal(rotation[:,:3,3],reference[:,:3,3])
    quarter,_,_=write_reference(reference,innovation,torch.full((2,2),.25),diameter)
    torch.testing.assert_close(quarter[:,:3,3],.75*reference[:,:3,3]+.25*measurement[:,:3,3])
    for _ in range(150):
        innovation,_=write_innovation(quarter,measurement,diameter)
        quarter,_,_=write_reference(quarter,innovation,torch.full((2,2),.25),diameter)
    r=quarter[:,:3,:3];assert (r.transpose(-1,-2)@r-torch.eye(3)).abs().max()<2e-5
    assert (torch.linalg.det(r)-1).abs().max()<2e-5 and (quarter[:,2,3]>0).all()


def test_writer_cuts_measurement_gradient_but_keeps_memory_gradient_and_handles_invalid_target():
    reference,measurement,diameter=poses();reference.requires_grad_();measurement.requires_grad_();gates=torch.full((2,2),.1,requires_grad=True)
    innovation,_=write_innovation(reference,measurement,diameter)
    written,_,_=write_reference(reference,innovation,gates,diameter)
    (written[:,0,1].sum()+written[:,0,3].sum()).backward()
    assert measurement.grad is None
    assert torch.isfinite(reference.grad).all() and (gates.grad.abs().sum(0)>0).all()
    bad=measurement.detach().clone();bad[0,0,0]=float('nan');bad[1,2,3]=-.1
    innovation,valid=write_innovation(reference,bad,diameter)
    assert not valid.any() and torch.isfinite(innovation).all()
    out,_,_=write_reference(reference,innovation,torch.zeros_like(gates),diameter)
    assert torch.equal(out,reference)
