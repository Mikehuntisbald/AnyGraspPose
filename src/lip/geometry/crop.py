import torch
import torch.nn.functional as F


def project(x, k):
    p = x @ k.transpose(-1, -2)
    return p[..., :2] / p[..., 2:].clamp_min(1e-6)


def crop_matrix(vertices, base, k, size=224, expansion=2.):
    camera = vertices @ base[:3, :3].T + base[:3, 3]
    good = camera[:, 2] > .001
    if good.any():
        uv = project(camera[good], k)
        lo, hi = uv.amin(0), uv.amax(0)
        mid = (lo+hi)/2
        side = ((hi-lo).max()*expansion).clamp(64, 4096)
    else:
        mid = k[:2, 2]; side = k.new_tensor(640.)
    # Integer coordinates denote pixel centers; maps crop edges at -0.5 and N-0.5.
    scale = size/side
    a = torch.eye(3, device=k.device, dtype=k.dtype)
    a[0, 0] = a[1, 1] = scale
    a[:2, 2] = (size-1)/2 - scale*mid
    return a, a @ k


def crop_images(x, a, size=224, mode='bilinear'):
    h, w = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(size, device=x.device, dtype=a.dtype),
                           torch.arange(size, device=x.device, dtype=a.dtype), indexing='ij')
    uv = torch.stack((xx, yy, torch.ones_like(xx)), -1) @ torch.linalg.inv(a).T
    grid = torch.stack((2*(uv[..., 0]+.5)/w-1, 2*(uv[..., 1]+.5)/h-1), -1)
    return F.grid_sample(x, grid[None].expand(len(x), -1, -1, -1), mode=mode,
                         padding_mode='zeros', align_corners=False)


def geometry_channels(depth, rendered, xyz, diameter, zbase):
    valid = torch.isfinite(depth) & (depth > 0)
    silhouette = rendered > 0
    both = valid & silhouette
    norm = lambda x, mask: torch.where(mask, x/diameter, 0.).clamp(-2, 2)
    return torch.cat((norm(depth-zbase, valid), valid.float(),
                      norm(rendered-zbase, silhouette).expand_as(depth),
                      silhouette.float().expand_as(depth),
                      (xyz/diameter).expand(len(depth), -1, -1, -1),
                      norm(depth-rendered, both), both.float()), 1)
