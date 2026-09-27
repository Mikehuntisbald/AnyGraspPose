import torch
from lip.unified.flow_atlas_prior import aligned_canonical_prior


def test_forward_endpoints_place_actual_canonical_identity_and_keep_depth():
    fallback = torch.randn(1, 5, 16, 16)
    xyz = torch.tensor([[[.1, .2, .3], [-.2, .1, .4]]])
    uv = torch.tensor([[[3., 7.], [12., 2.]]])
    result, coverage = aligned_canonical_prior(xyz, uv, torch.ones(1, 2, dtype=torch.bool), fallback, radius=1.)
    torch.testing.assert_close(result[0, :3, 7, 3], xyz[0, 0])
    torch.testing.assert_close(result[0, :3, 2, 12], xyz[0, 1])
    assert coverage[0, 0, 7, 3] and not coverage[0, 0, 0, 0]
    assert torch.equal(result[:, 3:], fallback[:, 3:])
    assert torch.equal(result[0, :, 0, 0], fallback[0, :, 0, 0])


def test_missing_or_nonfinite_anchors_leave_fallback_unchanged():
    fallback = torch.randn(2, 5, 16, 16)
    xyz = torch.zeros(2, 3, 3)
    uv = torch.full((2, 3, 2), float('nan'))
    result, coverage = aligned_canonical_prior(xyz, uv, torch.ones(2, 3, dtype=torch.bool), fallback)
    assert not coverage.any() and torch.equal(result, fallback)
