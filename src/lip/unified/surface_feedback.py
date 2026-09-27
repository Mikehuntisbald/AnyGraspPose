"""Surface-preserving canonical correspondence evidence from JEPA recovery.

Samples actual pixels rather than averaging XYZ across a patch. Predicted
validity only gates evidence; it cannot make an unsupported patch attractive
by removing a mismatch penalty. No teacher or pose solver is used here.
"""
import torch


def surface_feedback(reference_xyz, recovered, sigma=.1, strength=2., chunk=32):
    if sigma <= 0 or strength < 0 or chunk < 1:
        raise ValueError('Positive sigma/chunk and nonnegative strength required')
    if recovered.shape[1] < 5 or recovered.shape[-2:] != (224, 224):
        raise ValueError('Expected 224px XYZ/depth/validity recovery')
    batch = recovered.shape[0]
    # Sixteen actual raster samples per 14x14 patch, including both sides of
    # its center. Selection is fixed and independent of labels/confidence.
    offsets = torch.tensor([1, 5, 8, 12], device=recovered.device)
    pixels = recovered.float().reshape(batch, -1, 16, 14, 16, 14)
    pixels = pixels.index_select(3, offsets).index_select(5, offsets)
    pixels = pixels.permute(0, 2, 4, 3, 5, 1).reshape(batch, 256, 16, -1)
    xyz = pixels[..., :3]
    finite = torch.isfinite(xyz).all(-1) & torch.isfinite(pixels[..., 4])
    trust = torch.where(finite, pixels[..., 4].sigmoid(), 0.).detach()
    xyz = torch.where(finite[..., None], xyz, 0.)
    parts = []
    for start in range(0, reference_xyz.shape[1], chunk):
        source = reference_xyz[:, start:start + chunk].float()
        distance2 = (source[:, :, None, None] - xyz[:, None]).square().sum(-1)
        similarity = (-distance2 / (2 * sigma**2)).exp() * trust[:, None]
        # A patch must contain at least one compatible surface sample. Taking
        # the maximum preserves separate surfaces, unlike mean coordinates.
        parts.append(strength * similarity.max(-1).values)
    return torch.cat(parts, 1)
