import torch
from lip.unified.dpt_surface import DPTSurfaceHead


def test_completed_latent_is_only_input_for_all_dense_scales():
    torch.manual_seed(64)
    head=DPTSurfaceHead(final_only=True)
    levels=[torch.randn(1,256,256,requires_grad=True) for _ in range(4)]
    valid=torch.ones(1,256,dtype=torch.bool)
    result=head(levels,valid)
    alternate=[torch.randn_like(x)*5 for x in levels[:3]]+[levels[-1]]
    assert torch.equal(result,head(alternate,valid))
    grad=torch.autograd.grad(result.square().mean(),levels,allow_unused=True)
    assert all(x is None for x in grad[:3])
    assert torch.isfinite(grad[-1]).all() and grad[-1].norm()>0
    shifted=levels[:3]+[levels[-1].reshape(1,16,16,256).roll(1,2).reshape_as(levels[-1])]
    assert not torch.equal(result,head(shifted,valid))


def test_legacy_route_and_parameter_schema_remain_available():
    torch.manual_seed(64)
    old=DPTSurfaceHead();new=DPTSurfaceHead(final_only=True)
    new.load_state_dict(old.state_dict(),strict=True)
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    levels=[torch.randn(1,256,256) for _ in range(4)];valid=torch.ones(1,256,dtype=torch.bool)
    baseline=old(levels,valid)
    alternate=[levels[0].roll(1,1)]+levels[1:]
    assert not torch.equal(baseline,old(alternate,valid))
    # When all input states are equal, both modes perform the same operations.
    same=[levels[-1]]*4
    assert torch.equal(old(same,valid),new(same,valid))
