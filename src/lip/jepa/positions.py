import torch
from torch import nn
from torch.nn import functional as F


def patch_centers(device=None, dtype=torch.float32):
    a = (torch.arange(16, device=device, dtype=dtype) + .5) * 14 - .5
    y, x = torch.meshgrid(a, a, indexing='ij')
    return torch.stack((x, y), -1).reshape(256, 2)


def source_coordinates(affine, size_wh):
    xy = patch_centers(affine.device, affine.dtype)
    homogeneous = F.pad(xy, (0, 1), value=1)
    source = torch.linalg.solve(affine, homogeneous.T.expand(len(affine), -1, -1)).transpose(1, 2)[..., :2]
    return 2 * (source + .5) / size_wh[:, None] - 1


def sample_features(features, xy_crop):
    """224 pixel coordinates sampled on a stride-14 map; align_corners=False."""
    grid = 2 * (xy_crop.float() + .5) / 224 - 1
    return F.grid_sample(features.transpose(1, 2).reshape(-1, features.shape[-1], 16, 16).float(),
                         grid[:, :, None], align_corners=False).squeeze(-1).transpose(1, 2)


class PositionEncoding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.position = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.prompt = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, packet, last_time=None):
        source = source_coordinates(packet.A_image_to_crop, packet.source_size_wh)
        crop = 2 * (patch_centers(source.device) + .5) / 224 - 1
        pos = self.position(torch.cat((crop.expand(len(source), -1, -1), source), -1))
        dt = torch.zeros_like(packet.timestamp_s) if last_time is None else packet.timestamp_s - last_time
        return pos, self.prompt(packet.track_prompt / 224), self.time(dt.float().clamp(0, 60)[:, None]), source
