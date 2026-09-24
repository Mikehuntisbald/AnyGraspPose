import torch
from lip.jepa.losses import normalize
from lip.unified.local_structure import difference_terms, local_correspondence, local_structure_loss


def sample():
    torch.manual_seed(41)
    t = normalize(torch.randn(2, 64, 12))
    q = torch.ones(2, 64)
    return t, q


def test_exact_mean_and_spatial_permutation():
    t, q = sample()
    assert difference_terms(t, t, q, q)[0] == 0
    assert abs(float(local_correspondence(t, t, q, q))) < 1e-6
    for p in (t.mean(1, keepdim=True).expand_as(t), t.roll(2, 1)):
        assert difference_terms(p, t, q, q)[0] > .1
        assert local_correspondence(p, t, q, q) > .1


def test_common_component_cancels_in_difference_space():
    t, q = sample(); offset = torch.randn(2, 1, 12)*5
    # Additive invariance is after per-token normalization, not before it.
    assert difference_terms(t+offset, t, q, q)[0] < 1e-6
    assert abs(float(local_correspondence(t+offset, t, q, q))) < 1e-5


def test_only_hidden_predictions_receive_gradients_and_teacher_detaches():
    t, q = sample(); t.requires_grad_(); q[:, :32] = 0
    for function in (lambda *a: difference_terms(*a)[0], local_correspondence):
        p = torch.randn_like(t, requires_grad=True)
        function(p, t, q, torch.ones_like(q)).backward()
        assert p.grad[:, :32].abs().sum() == 0
        assert p.grad[:, 32:].abs().sum() > 0
        assert t.grad is None
        # No loss reads visible prediction values, even if arbitrarily corrupted.
        changed = p.detach().clone(); changed[:, :32] += 100
        assert torch.allclose(function(p, t, q, torch.ones_like(q)),
                              function(changed, t, q, torch.ones_like(q)))


def test_empty_flat_singleton_finite_and_no_fake_negatives():
    t, q = sample(); p = torch.randn_like(t, requires_grad=True)
    for weight in (q*0, torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), 64)):
        values = (difference_terms(p, t, weight, weight)[0], local_correspondence(p, t, weight, weight))
        assert all(torch.isfinite(x) and x == 0 for x in values)
    flat = t[:, :1].expand_as(t)
    assert difference_terms(flat, flat, q, q)[0] == 0
    assert abs(float(local_correspondence(flat, flat, q, q))) < 1e-6


def test_long_edges_do_not_cross_holes_and_no_grid_wrap():
    t, q = sample(); valid = q*0
    valid[:, 0] = 1; valid[:, 4] = 1
    assert difference_terms(t.roll(1, 1), t, valid, valid, scales=(4,))[0] == 0
    valid = q*0; valid[:, 7:9] = 1
    assert difference_terms(t.roll(1, 1), t, valid, valid, scales=(1,))[0] == 0


def test_source_isolation_and_equal_layer_gradients():
    from types import SimpleNamespace
    t, q = sample(); hidden = q.clone(); hidden[:, :32] = 0
    p = torch.randn_like(t)
    out = dict(f_mid_predicted=p.clone().requires_grad_(), f_predicted=p.clone().requires_grad_())
    target = SimpleNamespace(real_mid=t, real_last=t, proxy_mid=t+2, proxy_last=t+2,
        hidden_real_weight=hidden, visible_weight=q-hidden, proxy_weight=q*0, support_label=q)
    weights = dict(cad_feature=.5, feature_mid_weight=.625, feature_last_weight=.625,
                   local_difference=.2, local_correspondence=.05)
    loss, parts = local_structure_loss(out, target, weights); loss.backward()
    assert torch.allclose(out['f_mid_predicted'].grad, out['f_predicted'].grad)
    assert not out['f_predicted'].grad[:, :32].any()
    assert parts['local_difference_proxy_mid'] == 0


def test_local_losses_reach_shared_patch_latent():
    from test_recovered_relation import sample as model_sample
    from lip.unified.reconstruction_only import configure_reconstruction_only
    _, model, obs = model_sample(); configure_reconstruction_only(model)
    output, _ = model(obs); p = normalize(output['f_predicted'])
    t = torch.randn_like(p); q = torch.zeros(1, 256); q[:, 64:128] = 1
    for loss in (difference_terms(p, t, q, torch.ones_like(q))[0], local_correspondence(p, t, q, torch.ones_like(q))):
        grad = torch.autograd.grad(loss, output['patch_latent'], retain_graph=True)[0]
        assert grad.isfinite().all() and grad.abs().sum() > 0
