import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from lip.geometry.so3 import update, original_pose


def position(x, width):
    freq = torch.exp(torch.arange(width//2, device=x.device).float()*(-9.21034037/max(1, width//2-1)))
    v = x.float()[..., None]*freq
    return torch.cat((v.sin(), v.cos()), -1)


def spatial_position(side, device):
    yy, xx = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing='ij')
    return torch.cat((position(xx, 128), position(yy, 128)), -1).reshape(side*side, 256)


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(),
                                 nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c))
    def forward(self, x): return x+torch.nn.functional.gelu(self.net(x))


class CrossBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.qnorm, self.knorm, self.fnorm = [nn.LayerNorm(256) for _ in range(3)]
        self.attn = nn.MultiheadAttention(256, 8, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Dropout(dropout), nn.Linear(1024, 256), nn.Dropout(dropout))
    def forward(self, q, kv):
        kv = self.knorm(kv)
        q = q+self.attn(self.qnorm(q), kv, kv, need_weights=False)[0]
        return q+self.ff(self.fnorm(q))


class Tracker(nn.Module):
    def __init__(self, pretrained=True, dropout=.1):
        super().__init__()
        r = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.rgb = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.rgb_proj = nn.Conv2d(1024, 256, 1)
        layers=[]; prev=9
        for c in (32, 64, 128, 256):
            layers.extend([nn.Conv2d(prev, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.GELU(), Residual(c)]); prev=c
        self.geometry = nn.Sequential(*layers)
        self.fusion = nn.ModuleList([CrossBlock(dropout) for _ in range(2)])
        self.state = nn.Sequential(nn.Linear(24, 256), nn.GELU(), nn.Linear(256, 256))
        layer = nn.TransformerEncoderLayer(256, 8, 1024, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 4, norm=nn.LayerNorm(256), enable_nested_tensor=False)
        self.token_type = nn.Parameter(torch.zeros(3, 256))
        self.readout = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.normal_(self.readout, std=.02)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 6))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
        return self

    def forward(self, rgb, geometry, history_state, base_state, time_offsets_sec,
                frame_valid, history_pose_valid, T_base_centered, object_diameter_m, mesh_center):
        b, l = rgb.shape[:2]
        x = self.rgb_proj(self.rgb(rgb.flatten(0, 1)))
        g = self.geometry(geometry.flatten(0, 1))
        side=x.shape[-1]; pos=spatial_position(side, x.device).to(x.dtype)
        x=x.flatten(2).transpose(1, 2)+pos; g=g.flatten(2).transpose(1, 2)+pos
        for block in self.fusion: x=block(x, g)
        x = torch.nn.functional.adaptive_avg_pool2d(x.transpose(1, 2).reshape(b*l, 256, side, side), 4)
        x=x.flatten(2).transpose(1, 2).reshape(b, l, 16, 256)
        x=x+spatial_position(4, x.device).to(x.dtype)+self.token_type[0]
        # Defense in depth: caller cannot accidentally use unavailable current poses.
        hist=history_state*history_pose_valid[..., None]
        s=self.state(torch.cat((hist, base_state[:, None].expand(-1, l, -1)), -1))
        tokens=torch.cat((x, (s+self.token_type[1])[:, :, None]), 2)
        tokens=tokens+position(time_offsets_sec, 256)[:, :, None].to(tokens.dtype)
        tokens=tokens.flatten(1, 2)
        tokens=torch.cat((tokens, self.readout.expand(b, -1, -1)+self.token_type[2]), 1)
        frame_ids=torch.cat((torch.arange(l, device=x.device).repeat_interleave(17), torch.tensor([l-1], device=x.device)))
        causal=frame_ids[None, :] > frame_ids[:, None]
        keys=torch.cat((frame_valid.repeat_interleave(17, -1), torch.ones(b, 1, device=x.device, dtype=torch.bool)), -1)
        blocked=causal[None] | ~keys[:, None, :]
        # A padded query needs one finite attention entry; real queries never see it.
        invalid=~keys
        eye=torch.eye(len(frame_ids), device=x.device, dtype=torch.bool)
        blocked=blocked & ~(invalid[:, :, None] & eye[None])
        mask=blocked.repeat_interleave(8, 0)
        z=self.temporal(tokens, mask=mask)[:, -1]
        delta=self.head(z).float()
        with torch.autocast(x.device.type, enabled=False):
            pose=update(T_base_centered.float(), delta[:, :3], delta[:, 3:], object_diameter_m.float())
            original=original_pose(pose, mesh_center.float())
        return dict(pose_centered=pose, pose_original=original, delta_rotvec=delta[:, :3],
                    delta_center_norm=delta[:, 3:], latent=z)
