"""Optional branch. Channels audited from lip.geometry.crop.geometry_channels.

0 observed relative depth/d, 1 observed valid, 2 rendered relative depth/d,
3 rendered silhouette, 4:7 centered object XYZ/d, 7 depth residual/d,
8 observed-and-rendered validity. There are no normal channels in this renderer.
"""
from torch import nn
from torch.nn import functional as F

CHANNEL_CONTRACT = ['observed_relative_depth_over_d', 'observed_valid',
    'rendered_relative_depth_over_d', 'rendered_silhouette',
    'centered_object_x_over_d', 'centered_object_y_over_d', 'centered_object_z_over_d',
    'observed_minus_rendered_depth_over_d', 'both_depth_valid']


class GeometryAdapter(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(9, 64, 7, 7), nn.GroupNorm(8, 64), nn.GELU(),
                                     nn.Conv2d(64, dim, 4, 4), nn.GroupNorm(8, dim), nn.GELU())

    def forward(self, packet):
        tokens = self.network(packet.channels).flatten(2).transpose(1, 2)
        valid = F.adaptive_max_pool2d(packet.channels[:, 3:4], 8).flatten(1) > 0
        return tokens, valid & packet.available[:, None], packet.quality * packet.available
