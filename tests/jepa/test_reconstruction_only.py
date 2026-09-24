from dataclasses import replace
import torch
from torch import nn
from lip.unified.features import Scene, TeacherTargets, build_teachers
from lip.unified.losses import DEFAULT_WEIGHTS
from lip.unified.optimized_training import make_optimizer
from lip.unified.reconstruction_only import configure_reconstruction_only, is_pose_parameter, recovery_objective
from test_recovered_relation import sample


def test_recovery_gradients_ignore_pose_and_optimizer_excludes_frozen_readout():
    _, model, obs = sample()
    configure_reconstruction_only(model)
    out, _ = model(obs)
    features = torch.randn(1, 256, 384); weights = torch.ones(1, 256)
    dense = torch.ones(1, 1, 224, 224, dtype=torch.bool)
    real = weights.clone(); real[:, 128:] = 0; proxy = 1 - real
    target = TeacherTargets(features, features, features, features, proxy, real, proxy, weights * 0, proxy,
        weights, weights, torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224),
        torch.zeros(1, 3, 224, 224), dense.float(), dense.float(), dense, dense, ~dense,
        dense.float(), dense, ~dense)
    # Poisoned pose tensors must never be consulted by the training objective.
    out['pose_centered'] = torch.full_like(out['pose_centered'], float('nan'), requires_grad=True)
    loss, metrics = recovery_objective(out, target, None, None, None, DEFAULT_WEIGHTS)
    loss.backward()
    assert torch.isfinite(loss) and not metrics[:4].any()
    assert out['pose_centered'].grad is None
    assert model.core.blocks[0].spatial.q.weight.grad.norm() > 0
    assert model.surface_head[-1].weight.grad.norm() > 0
    assert model.core.feature_last.weight.grad.norm() > 0
    assert all(not p.requires_grad and p.grad is None for n, p in model.named_parameters() if is_pose_parameter(n))
    optimizer = make_optimizer(model, dict(training=dict(learning_rates=dict(new=1e-4, predictor=1e-5, pose=3e-6), weight_decay=.01)), fused=False)
    optimized = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert optimized == {id(p) for p in model.parameters() if p.requires_grad}


def test_impossible_depth_is_unknown_without_erasing_rgb_or_substituting_cad():
    class Encoder(nn.Module):
        def forward(self, x):
            f = torch.nn.functional.avg_pool2d(x, 14, 14).flatten(2).transpose(1, 2).repeat(1, 1, 128)
            return f, f
    pose = torch.eye(4); pose[2, 3] = .8
    depth = torch.full((1, 1, 224, 224), .8); depth[:, :, 70:140, 70:140] = 12.
    silhouette = torch.ones(224, 224, dtype=torch.bool)
    def render(*args):
        return dict(rgb=torch.ones(3, 224, 224) * .8, mask=silhouette,
                    depth=torch.ones(1, 224, 224) * .8, xyz=torch.zeros(3, 224, 224))
    scene = Scene(torch.rand(1, 3, 224, 224), depth, {}, torch.eye(3),
        torch.tensor([[1000., 0., 112.], [0., 1000., 112.], [0., 0., 1.]]),
        torch.ones(1, 1, 224, 224, dtype=torch.bool), pose, .2, torch.zeros(3), torch.zeros(24), 0., 'a', (224, 224), {'appearance': {}})
    args = (Encoder(), [scene], [pose], [silhouette[None]], [scene.bounds], render)
    old = build_teachers(*args)
    new = build_teachers(*args, real_geometry_max_radius_d=1.)
    assert old.geometry_real_weight[:, :, 80:130, 80:130].all()
    assert not new.geometry_real_weight[:, :, 70:140, 70:140].any()
    assert torch.isnan(new.geometry_valid_label[:, :, 70:140, 70:140]).all()
    assert new.geometry_real_weight[:, :, 30:50, 30:50].all()
    assert torch.equal(old.hidden_real_weight, new.hidden_real_weight)
    assert torch.equal(old.real_last, new.real_last)
    assert not new.geometry_proxy_weight.any()
    assert torch.equal(old.surface_depth_m, new.surface_depth_m)
