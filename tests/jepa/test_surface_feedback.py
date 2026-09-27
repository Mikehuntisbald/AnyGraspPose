import torch
from lip.unified.surface_feedback import surface_feedback


def test_separate_surfaces_do_not_create_a_phantom_mean_surface():
    recovered = torch.zeros(1, 5, 224, 224)
    recovered[:, 4] = 12
    recovered[:, 0, :, :7] = -.3
    recovered[:, 0, :, 7:14] = .3
    # Patch 1 is a real surface at zero; patch 0 averages to zero but has none.
    reference = torch.tensor([[[0., 0., 0.], [.3, 0., 0.]]])
    scores = surface_feedback(reference, recovered, sigma=.05)
    assert scores[0, 0, 1] > 1.99
    assert scores[0, 0, 0] < 1e-6
    assert scores[0, 1, 0] > 1.99


def test_invalid_geometry_cannot_supply_matching_evidence():
    recovered = torch.zeros(1, 5, 224, 224)
    recovered[:, 4] = -float('inf')
    recovered[:, :3, :14, :14] = float('nan')
    score = surface_feedback(torch.zeros(1, 3, 3), recovered)
    assert torch.isfinite(score).all() and not score.any()


def test_geometry_has_gradients_but_validity_cannot_game_feedback():
    recovered = torch.zeros(1, 5, 224, 224, requires_grad=True)
    reference = torch.tensor([[[.03, .02, .01]]], requires_grad=True)
    surface_feedback(reference, recovered).sum().backward()
    assert recovered.grad[:, :3].norm() > 0
    assert reference.grad.norm() > 0
    assert not recovered.grad[:, 4].any()
    assert torch.isfinite(recovered.grad).all()


def test_chunking_preserves_source_order_and_scores():
    torch.manual_seed(67)
    recovered = torch.randn(2, 5, 224, 224)
    reference = torch.randn(2, 7, 3)
    torch.testing.assert_close(surface_feedback(reference, recovered, chunk=2),
                               surface_feedback(reference, recovered, chunk=7), rtol=0, atol=0)
