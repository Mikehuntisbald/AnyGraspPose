import torch
from lip.unified.cad_transport import pixel_grid, sample_reference, transport_targets, CADTransport


def test_lookup_identity_and_fractional_gradient():
    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')
    image = (2*x+3*y).float()[None, None]
    uv = (pixel_grid(1, 8, 8, 'cpu')+.25).requires_grad_()
    value, mass = sample_reference(image, torch.ones_like(image, dtype=torch.bool), uv)
    torch.testing.assert_close(value[:, :, :-1, :-1], image[:, :, :-1, :-1]+1.25)
    value[:, :, 2:5, 2:5].sum().backward()
    torch.testing.assert_close(uv.grad[:, 0, 2:5, 2:5], torch.full((1, 3, 3), 2.))
    torch.testing.assert_close(uv.grad[:, 1, 2:5, 2:5], torch.full((1, 3, 3), 3.))


def test_empty_and_outside_reference():
    image = torch.full((1, 4, 8, 8), float('nan'))
    uv = (pixel_grid(1, 8, 8, 'cpu')+100).requires_grad_()
    value, mass = sample_reference(image, torch.zeros(1, 1, 8, 8, dtype=torch.bool), uv)
    assert torch.isfinite(value).all() and not value.any() and not mass.any()
    value.sum().backward()
    assert torch.isfinite(uv.grad).all()


def test_projection_label_and_nonzero_pose_flow():
    grid = pixel_grid(1, 224, 224, 'cpu')
    xyz = torch.cat(((grid-111.5)/224., torch.zeros(1, 1, 224, 224)), 1)
    g = torch.zeros(1, 9, 224, 224); g[:, 3] = 1; g[:, 4:7] = xyz
    base = torch.eye(4)[None]; base[:, 2, 3] = 1
    k = torch.tensor([[[224., 0., 111.5], [0., 224., 111.5], [0., 0., 1.]]])
    valid = torch.ones(1, 256, dtype=torch.bool)
    target = transport_targets(xyz, g, base, torch.ones(1), k, valid)
    assert target['supported'].all()
    torch.testing.assert_close(target['flow'], torch.zeros_like(target['flow']), atol=2e-5, rtol=0)
    # The same canonical point moves right under a translated estimated pose.
    base[:, 0, 3] = .02
    shifted = transport_targets(xyz, g, base, torch.ones(1), k, valid)
    torch.testing.assert_close(shifted['flow'][:, 0], torch.full((1, 224, 224), 4.48), atol=3e-5, rtol=0)


def test_control_parity_and_cad_dropout():
    torch.manual_seed(42)
    head = CADTransport(enabled=False)
    dense = torch.randn(1, 16, 224, 224, requires_grad=True)
    fallback = torch.randn(1, 5, 224, 224, requires_grad=True)
    geometry = torch.randn(1, 9, 224, 224)
    valid = torch.ones(1, 256, dtype=torch.bool)
    result, _ = head(dense, fallback, geometry, valid)
    assert torch.equal(result, fallback)
    head.enabled = True
    result, metrics = head(dense, fallback, geometry, valid & False)
    assert torch.equal(result, fallback) and not metrics['transport_gate'].any()
    result.sum().backward()
    assert torch.equal(fallback.grad, torch.ones_like(fallback))


def test_reference_extension_exact_initial_function_and_gradient():
    torch.manual_seed(42)
    old=CADTransport();new=CADTransport(reference_conditioned=True)
    # Exercise a trained nonzero output head; a zero head would hide mistakes.
    torch.nn.init.normal_(old.head[-1].weight,std=.01)
    state=old.state_dict();weight=torch.zeros_like(new.head[0].weight)
    weight[:,:16]=state['head.0.weight'];state['head.0.weight']=weight
    new.load_state_dict(state)
    dense=torch.randn(1,16,224,224);fallback=torch.randn(1,5,224,224)
    geometry=torch.randn(1,9,224,224);geometry[:,3]=1
    valid=torch.ones(1,256,dtype=torch.bool)
    a,_=old(dense,fallback,geometry,valid);b,_=new(dense,fallback,geometry,valid)
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    b[:,:4].square().mean().backward()
    assert new.head[0].weight.grad[:,16:].norm()>0
