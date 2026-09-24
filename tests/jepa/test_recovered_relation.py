from dataclasses import replace
import io
import pytest
import torch
from torch import nn
from lip.unified.recovered_relation import (
    RecoveredGeometryRelation, RecoveredRelationTracker, cad_correspondence,
    initialize_from_supported)
from lip.unified.supported_history import SupportedHistoryTracker
from test_unified import observation


def sample():
    torch.manual_seed(42)
    point = nn.Identity(); point.feature_dim = 12
    parent = SupportedHistoryTracker(nn.Identity(), point).eval()
    model = RecoveredRelationTracker(nn.Identity(), point).eval()
    initialize_from_supported(model, parent.state_dict())
    image = torch.randn(1, 9, 224, 224) * .1
    image[:, [1, 3, 8]] = 1
    obs = replace(observation(), geo=torch.randn(1, 12), geometry_image=image)
    return parent, model, obs


def test_soft_correspondence_is_cross_position_and_permutation_invariant():
    source = torch.tensor([[[.2, .1, 0.]]])
    reference = torch.tensor([[[-.5, 0., 0.], [.2, .1, 0.], [.8, .5, .5]]])
    depth = torch.tensor([[.3, .1, -.2]]); valid = torch.ones(1, 3, dtype=torch.bool)
    xyz, z, _, _ = cad_correspondence(source, reference, depth, valid)
    assert torch.allclose(xyz, source, atol=1e-5) and torch.allclose(z, depth[:, 1:2], atol=1e-5)
    other = cad_correspondence(source, reference.flip(1), depth.flip(1), valid)
    assert torch.allclose(xyz, other[0], atol=1e-6)


def test_far_observation_cannot_select_masked_cad_and_empty_cad_is_finite():
    source = torch.tensor([[[100., 100., 100.]]], requires_grad=True)
    reference = torch.tensor([[[.3, .2, .1], [100., 100., 100.]]])
    depth = torch.tensor([[.2, 10.]])
    valid = torch.tensor([[True, False]])
    xyz, z, _, _ = cad_correspondence(source, reference, depth, valid)
    assert torch.equal(xyz, reference[:, :1]) and torch.equal(z, depth[:, :1])
    empty = cad_correspondence(source, reference, depth, valid & False)
    assert all(torch.isfinite(v).all() for v in empty)
    assert not any(v.any() for v in empty)


def test_pose_gradient_reaches_xyz_depth_validity_and_shared_patch():
    _, model, obs = sample()
    captured = []
    h = model.surface_head.register_forward_hook(lambda _m, _a, out: captured.append(out))
    out, _ = model(obs); h.remove()
    gs, gp = torch.autograd.grad(out['pose_centered'].square().mean(),
                                (captured[0], out['patch_latent']), retain_graph=True)
    gs = gs.reshape(1, 256, 14, 14, 5)
    assert all(torch.isfinite(gs[..., i]).all() and gs[..., i].abs().sum() > 0 for i in range(5))
    assert gp.abs().sum() > 0
    exposed = torch.autograd.grad(out['pose_centered'].square().mean(),
        (out['surface_xyz'], out['surface_depth_residual'], out['geometry_valid_logits']))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in exposed)
    read = []
    h = model.object_attn.register_forward_pre_hook(lambda _m, args: read.append(args[1]))
    second, _ = model(obs); h.remove()
    assert read[0] is second['pose_relation_latent']


def test_recovered_geometry_intervention_changes_pose_without_changing_memory():
    _, model, obs = sample()
    with torch.no_grad():
        a, ma = model(obs)
        def alter(_m, _a, value):
            changed = value.clone().reshape(1, 256, 14, 14, 5)
            changed[..., :4] += .3
            return changed.reshape_as(value)
        h = model.surface_head.register_forward_hook(alter)
        b, mb = model(obs); h.remove()
    assert torch.equal(a['patch_latent'], b['patch_latent'])
    assert not torch.equal(a['pose_centered'], b['pose_centered'])
    for kind in ('objects', 'contexts'):
        for x, y in zip(getattr(ma, kind), getattr(mb, kind)):
            assert all(torch.equal(getattr(x, n), getattr(y, n)) for n in x.__dataclass_fields__)


def relation_input(visibility=-10., depth=True):
    patch = torch.randn(1, 256, 256)
    surface = torch.randn(1, 5, 224, 224)
    geometry = torch.randn(1, 9, 224, 224) * .1
    geometry[:, 1] = float(depth); geometry[:, 3] = 1
    return (patch, surface, geometry, torch.eye(4)[None], torch.ones(1, 256, dtype=torch.bool),
            torch.ones(1, 256, dtype=torch.bool), torch.full((1, 256), visibility),
            torch.randn(1, 256, 3), torch.full((1, 256), depth, dtype=torch.bool))


def test_trusted_measurement_takes_precedence_over_completion():
    module = RecoveredGeometryRelation(); args = relation_input(visibility=100.)
    a, meta = module(*args)
    altered = list(args); altered[1] = args[1] + 10
    b, _ = module(*altered)
    assert torch.equal(a, b) and not meta['relation_completed_weight'].any()
    assert (meta['relation_measured_weight'] == 1).all()


def test_no_depth_has_no_fabricated_measurement_and_completion_is_bounded():
    module = RecoveredGeometryRelation(); args = relation_input(depth=False)
    a, meta = module(*args)
    altered = list(args); altered[7] = args[7] + 999
    b, _ = module(*altered)
    assert torch.equal(a, b) and not meta['relation_measured_weight'].any()
    assert (meta['relation_completed_weight'] <= .5).all()
    assert meta['relation_completed_weight'].sum() > 0 and torch.isfinite(a).all()


def test_invalid_completion_is_not_evidence_and_dropped_cad_cannot_leak():
    module = RecoveredGeometryRelation(); args = list(relation_input(depth=False))
    args[1][:, 4] = -1000
    a, meta = module(*args)
    assert torch.equal(a, args[0]) and not meta['relation_completed_weight'].any()
    args[1][:, 4] = 0; args[5].zero_()
    a, _ = module(*args)
    args[2][:, 2:] += 100
    b, _ = module(*args)
    assert torch.equal(a, b)


def test_exact_parent_transfer_and_explicit_checkpoint_boundary():
    parent, model, obs = sample()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in parent.state_dict().items())
    with pytest.raises(RuntimeError): model.load_state_dict(parent.state_dict(), strict=True)
    with pytest.raises(ValueError): initialize_from_supported(model, {'bad': torch.zeros(1)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    def update(m, opt):
        opt.zero_grad(set_to_none=True)
        out, _ = m(obs); out['pose_centered'].square().mean().backward(); opt.step()
    update(model, optimizer)
    buffer = io.BytesIO(); torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, buffer)
    buffer.seek(0); record = torch.load(buffer, weights_only=False)
    _, clone, _ = sample(); clone.load_state_dict(record['model'], strict=True)
    opt2 = torch.optim.AdamW(clone.parameters(), lr=1e-4); opt2.load_state_dict(record['optimizer'])
    update(model, optimizer); update(clone, opt2)
    assert all(torch.equal(v, clone.state_dict()[k]) for k, v in model.state_dict().items())
