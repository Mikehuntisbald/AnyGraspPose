"""Frozen diagnostic: use flow-aligned CAD identities to locate atlas search.

This changes the canonical lookup prior, not the depth prediction. It is not a
trained decoder and the hard nearest-anchor assignment is not differentiable.
"""
import torch


@torch.no_grad()
def aligned_canonical_prior(reference_xyz, uv, available, fallback, radius=14.):
    batch, _, height, width = fallback.shape
    y, x = torch.meshgrid(torch.arange(height, device=uv.device),
                         torch.arange(width, device=uv.device), indexing='ij')
    grid = torch.stack((x, y), -1).float().reshape(-1, 2)
    safe = available & torch.isfinite(uv).all(-1) & torch.isfinite(reference_xyz).all(-1)
    endpoints = torch.where(safe[..., None], uv.float(), 0.)
    xyz = torch.where(safe[..., None], reference_xyz.float(), 0.)
    indices=[];distances=[]
    for start in range(0, len(grid), 2048):
        distance = (grid[None, start:start+2048, None] - endpoints[:, None]).square().sum(-1)
        value, index = distance.masked_fill(~safe[:, None], float('inf')).min(-1)
        indices.append(index);distances.append(value)
    index = torch.cat(indices, 1)
    distance = torch.cat(distances, 1)
    prior = xyz[torch.arange(batch, device=uv.device)[:, None], index]
    prior = prior.transpose(1, 2).reshape(batch, 3, height, width)
    coverage = (distance <= radius**2).reshape(batch, 1, height, width)
    result = fallback.clone()
    result[:, :3] = torch.where(coverage, prior.to(fallback.dtype), fallback[:, :3])
    return result, coverage
