import torch
import pytest
from lip.geometry.so3 import update, exp
from lip.models.pose_reference import PoseReferenceFeedback, reference_action, apply_reference_feedback
from test_direct_pose_residual import fixture


def test_reference_action_reconstructs_pose_including_near_pi_and_mesh_center():
    parent, features = fixture()
    reference = features['T_base_centered'].clone()
    reference[:, :3, :3] = exp(torch.tensor([[0., 0., torch.pi - 1e-5], [.5, -.2, .1]]))
    reference[:, :3, 3] += torch.tensor([.03, -.01, .04])
    action = reference_action(features['T_base_centered'], reference, features['object_diameter_m'])
    actual = apply_reference_feedback(parent, features, action, torch.ones(2, 2))
    torch.testing.assert_close(actual['pose_centered'], reference, atol=5e-7, rtol=1e-6)
    # Rotation feedback alone leaves the center untouched.
    rotation_only = apply_reference_feedback(parent, features, action, torch.tensor([[1., 0.], [1., 0.]]))
    assert torch.equal(rotation_only['pose_centered'][:, :3, 3], parent['pose_centered'][:, :3, 3])


def test_zero_feedback_is_exact_parent_and_can_learn_both_components():
    parent, features = fixture();head = PoseReferenceFeedback()
    reference = features['T_base_centered'].clone().requires_grad_()
    target = reference_action(features['T_base_centered'], reference, features['object_diameter_m'])
    coefficient = head(torch.randn(2, 256), torch.randn(2, 24), target, torch.ones(2), torch.arange(2.))
    assert not torch.count_nonzero(coefficient)
    out = apply_reference_feedback(parent, features, target, coefficient)
    for key in parent: assert torch.equal(out[key], parent[key])
    (out['pose_centered'][:, 0, 1].sum() + out['pose_centered'][:, 0, 3].sum()).backward()
    assert torch.isfinite(head.readout[-1].weight.grad).all()
    assert (head.readout[-1].weight.grad.abs().sum(1) > 0).all()
    assert reference.grad is None


def test_signed_feedback_has_explicit_range_and_valid_shapes():
    head = PoseReferenceFeedback();head.readout[-1].bias.data.copy_(torch.tensor([-1., 1.]))
    inputs = (torch.zeros(2, 256), torch.zeros(2, 24), torch.zeros(2, 6), torch.ones(2), torch.ones(2))
    result = head(*inputs)
    assert ((result > -1) & (result < 1)).all() and (result[:, 0] < 0).all() and (result[:, 1] > 0).all()
    with pytest.raises(ValueError): head(inputs[0][:, None], *inputs[1:])
