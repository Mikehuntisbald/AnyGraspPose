import torch
from lip.unified.geometry_priority import backward_primary_geometry


def test_head_full_gradient_and_restorer_budget():
    shared=torch.nn.Parameter(torch.tensor([1.,2.]));head=torch.nn.Parameter(torch.tensor([3.,4.]))
    patch=shared*2
    geometry=patch.square().sum()
    secondary=100*(patch*head).sum()
    geo_grad,=torch.autograd.grad(geometry,shared,retain_graph=True)
    sec_grad,head_grad=torch.autograd.grad(secondary,(shared,head),retain_graph=True)
    stats=backward_primary_geometry(geometry,secondary,[patch],[('core.blocks.0.weight',shared),('head.weight',head)],ratio=.5)
    torch.testing.assert_close(head.grad,head_grad)
    torch.testing.assert_close(shared.grad,geo_grad+stats['restoration_secondary_scale']*sec_grad)
    assert stats['effective_secondary_patch_ratio']<=.500001


def test_no_reweight_when_secondary_small_and_hooks_removed():
    shared=torch.nn.Parameter(torch.tensor([1.,2.]));patch=shared*2
    geometry=100*patch.square().sum();secondary=patch.sum()
    expected,=torch.autograd.grad(geometry+secondary,shared,retain_graph=True)
    stats=backward_primary_geometry(geometry,secondary,[patch],[('surface_head.weight',shared)])
    assert stats['restoration_secondary_scale']==1
    torch.testing.assert_close(shared.grad,expected)
    shared.grad=None;(100*shared).sum().backward()
    torch.testing.assert_close(shared.grad,torch.full_like(shared,100.))


def test_appearance_heads_not_suppressed():
    shared=torch.nn.Parameter(torch.tensor([1.,2.]));appearance=torch.nn.Parameter(torch.tensor([2.,3.]))
    patch=shared*2;geometry=patch.square().sum();secondary=100*(appearance*patch).sum()
    expected,=torch.autograd.grad(secondary,appearance,retain_graph=True)
    backward_primary_geometry(geometry,secondary,[patch],[('core.blocks.0.weight',shared),('core.feature_mid.weight',appearance)])
    torch.testing.assert_close(appearance.grad,expected)
