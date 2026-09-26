from types import SimpleNamespace
import torch
from lip.unified.flow_reconstruction import (FlowReconstruction, template_points,
    patch_grid, flow_labels, flow_reconstruction_loss)


def fixture():
    torch.manual_seed(56)
    geometry = torch.zeros(1, 9, 224, 224)
    y, x = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')
    geometry[:, 1] = geometry[:, 3] = 1
    geometry[:, 4] = (x-111.5)/224
    geometry[:, 5] = (y-111.5)/224
    valid = torch.ones(1, 256, dtype=torch.bool)
    patch = torch.randn(1, 256, 256, requires_grad=True)
    observed = torch.randn_like(patch)
    cad = torch.randn_like(patch)
    return FlowReconstruction(), patch, observed, cad, geometry, valid


def test_reference_is_actual_raster_surface_not_patch_mean():
    _, _, _, _, geometry, valid = fixture()
    geometry[:, 3, :14, :14] = 0
    geometry[:, 3, 1, 2] = 1
    reference = template_points(geometry, valid)
    assert reference['uv'][0, 0].tolist() == [2., 1.]
    torch.testing.assert_close(reference['xyz'][0, 0], geometry[0, 4:7, 1, 2])
    valid[:, 0] = False
    assert not template_points(geometry, valid)['available'][0, 0]


def test_reconstruction_consumes_flow_and_flow_consumes_reconstructed_geometry():
    model, patch, observed, cad, geometry, valid = fixture()
    reference = template_points(geometry, valid)
    updated, first = model(patch, observed, cad, geometry, valid, reference)
    grad, = torch.autograd.grad(updated.square().mean(), first['uv'], retain_graph=True)
    assert grad.norm() > 0 and torch.isfinite(grad).all()
    recovered = geometry[:, 4:7].clone()
    recovered = torch.cat((recovered, torch.zeros_like(recovered[:, :2])), 1).requires_grad_()
    _, second = model(updated, observed, cad, geometry, valid, reference, recovered)
    backward, = torch.autograd.grad(second['uv'].square().mean(), recovered)
    assert backward[:, :3].norm() > 0 and torch.isfinite(backward).all()
    model.disable_recovery_feedback = True
    _, no_feedback = model(updated, observed, cad, geometry, valid, reference, recovered)
    assert not torch.equal(second['uv'], no_feedback['uv'])
    model.disable_transport = True
    unchanged, _ = model(patch, observed, cad, geometry, valid, reference)
    assert torch.equal(unchanged, patch)


def test_missing_cad_and_invalid_crop_cannot_write_evidence():
    model, patch, observed, cad, geometry, valid = fixture()
    reference = template_points(geometry, torch.zeros_like(valid))
    updated, result = model(patch, observed, cad, geometry, valid, reference)
    assert torch.equal(updated, patch) and not result['aligned_mass'].any()
    assert torch.isfinite(result['uv']).all()
    empty, _ = model(patch, observed, cad, geometry, torch.zeros_like(valid), template_points(geometry, valid))
    assert torch.equal(empty, patch)


def test_projection_and_disjoint_real_proxy_flow_supervision():
    model, patch, observed, cad, geometry, valid = fixture()
    reference = template_points(geometry, valid)
    target = SimpleNamespace(camera_rotation=torch.eye(3)[None], camera_translation_d=torch.tensor([[0., 0., 1.]]),
        cad_geometry_valid=torch.ones(1, 1, 224, 224, dtype=torch.bool), cad_geometry_xyz=geometry[:, 4:7],
        cad_geometry_depth_m=torch.ones(1, 1, 224, 224), real_geometry_eligible=None,
        geometry_real_weight=torch.zeros(1, 1, 224, 224, dtype=torch.bool),
        geometry_proxy_weight=torch.zeros(1, 1, 224, 224, dtype=torch.bool))
    target.geometry_real_weight[:, :, :, 70:140] = True
    target.geometry_proxy_weight[:, :, :, 140:] = True
    visible = torch.zeros_like(target.geometry_real_weight); visible[:, :, :, :70] = True
    k = torch.tensor([[[224., 0., 111.5], [0., 224., 111.5], [0., 0., 1.]]])
    output = dict(flow_reference=reference)
    labels = flow_labels(output, target, k, torch.ones(1), visible)
    torch.testing.assert_close(labels['uv'], reference['uv'])
    assert not (labels['observed'] & labels['real']).any()
    assert not (labels['real'] & labels['proxy']).any()
    for name in ('observed', 'real', 'proxy'): assert labels[name].any()
    updated, result = model(patch, observed, cad, geometry, valid, reference)
    output.update(flow_rounds=[result, result], surface_xyz=updated)
    loss, _ = flow_reconstruction_loss(output, labels)
    loss.backward()
    assert patch.grad.norm() > 0 and model.endpoint[-1].weight.grad.norm() > 0
    # Changing supervision cannot affect a repeat student forward.
    labels['uv'] += 7
    again, _ = model(patch, observed, cad, geometry, valid, reference)
    assert torch.equal(updated, again)


def test_transport_gain_changes_write_without_changing_first_round_matches():
    model,patch,observed,cad,geometry,valid=fixture()
    reference=template_points(geometry,valid)
    _,baseline=model(patch,observed,cad,geometry,valid,reference)
    model.transport_gain=100.
    _,strong=model(patch,observed,cad,geometry,valid,reference)
    assert torch.equal(baseline['uv'],strong['uv'])
    torch.testing.assert_close(strong['write'],100*baseline['write'])
    model.disable_transport=True
    unchanged,_=model(patch,observed,cad,geometry,valid,reference)
    assert torch.equal(unchanged,patch)
