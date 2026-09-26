"""DPT-style multi-level dense geometry decoding from the unified JEPA only.

Four JEPA block outputs are reassembled into a spatial pyramid and refined
coarse-to-fine. No RGB/encoder skip or GT geometry enters this decoder.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FixedBilinear(nn.Module):
    """Separable bilinear resize with deterministic matmul backward on CUDA.

    Fixed 224px crops allow precomputing the align_corners=False interpolation
    operator. This avoids CUDA interpolate backward's atomic accumulation.
    """
    def __init__(self, incoming, outgoing):
        super().__init__()
        basis = torch.eye(incoming).reshape(incoming, 1, incoming)
        matrix = F.interpolate(basis, size=outgoing, mode='linear', align_corners=False)[:, 0].T
        self.register_buffer('matrix', matrix.contiguous(), persistent=False)

    def forward(self, value):
        matrix = self.matrix.to(value.dtype)
        return matrix @ value @ matrix.T


class ResidualUnit(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.GELU(), nn.Conv2d(width, width, 3, padding=1),
                                    nn.GELU(), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, value):
        return value + self.layers(value)


class Refine(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.skip = ResidualUnit(width)
        self.fused = ResidualUnit(width)
        self.project = nn.Conv2d(width, width, 1)

    def forward(self, value, skip):
        return self.project(self.fused(value + self.skip(skip)))


class DPTSurfaceHead(nn.Module):
    def __init__(self, width=64, final_only=False):
        super().__init__()
        self.final_only = bool(final_only)
        if width != 64:
            raise ValueError('Initial DPT experiment fixes pyramid width at 64')
        self.norms = nn.ModuleList([nn.LayerNorm(256) for _ in range(4)])
        self.projects = nn.ModuleList([nn.Conv2d(256, width, 1) for _ in range(4)])
        self.reassemble = nn.ModuleList([FixedBilinear(16, side) for side in (64, 32, 16, 8)])
        self.spatial = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in range(4)])
        self.coarse = ResidualUnit(width)
        self.refine = nn.ModuleList([Refine(width) for _ in range(3)])
        self.upsample = nn.ModuleList([FixedBilinear(side, 2*side) for side in (8, 16, 32)])
        self.output = nn.Sequential(
            nn.Conv2d(width, 32, 3, padding=1), nn.GELU(), FixedBilinear(64, 112),
            nn.Conv2d(32, 16, 3, padding=1), nn.GELU(), FixedBilinear(112, 224),
            nn.Conv2d(16, 5, 3, padding=1))
        # Signed XYZ/depth residuals and a raw validity logit, not positive depth.
        nn.init.normal_(self.output[-1].weight, std=.001)
        nn.init.zeros_(self.output[-1].bias)

    def dense_features(self, levels, valid):
        if len(levels) != 4:
            raise ValueError('DPT requires all four JEPA levels')
        if self.final_only:
            # All pyramid branches consume the completed current-stage latent.
            # The earlier JEPA states cannot bypass the flow/recovery updates.
            levels = (levels[-1],) * 4
        maps = []
        for tokens, norm, project, resize, spatial in zip(
                levels, self.norms, self.projects, self.reassemble, self.spatial):
            tokens = norm(torch.where(valid[..., None], tokens, 0.)) * valid[..., None]
            grid = tokens.transpose(1, 2).reshape(len(tokens), 256, 16, 16)
            maps.append(spatial(resize(project(grid))))
        value = self.coarse(maps[-1])
        for upsample, refine, skip in zip(self.upsample, self.refine, reversed(maps[:-1])):
            value = refine(upsample(value), skip)
        return self.output[:-1](value)

    def forward(self, levels, valid):
        return self.output[-1](self.dense_features(levels, valid)).float()


def enable_dpt_surface(model, config):
    if model.architecture_id not in ('stream_cad_surface_jepa_v12','stream_serial_completion_jepa_v21'):
        raise ValueError('DPT experiment requires the CAD-surface JEPA backbone')
    if config.get('kind') != 'dpt':
        raise ValueError('Expected explicit dpt surface decoder configuration')
    model.surface_head = DPTSurfaceHead(config.get('width', 64), config.get('final_only', False)).to(model.query.device)
    model.surface_decoder_kind = 'dpt'
    model.model_version = 'local-cad-dpt-rope3d-v17'


def migrate_dpt_surface(model, source):
    """Only the old independent-patch MLP is replaced; every shared tensor exact."""
    if not isinstance(model.surface_head, DPTSurfaceHead) or 'surface_head.3.weight' not in source:
        raise ValueError('Expected an explicit MLP-to-DPT migration')
    shared = {k:v for k,v in source.items() if not k.startswith('surface_head.')}
    status = model.load_state_dict(shared, strict=False)
    allowed = lambda key: key.startswith('surface_head.') or key == 'cad_surface.rope3d.gain'
    if status.unexpected_keys or any(not allowed(k) for k in status.missing_keys):
        raise ValueError('Unexpected state mismatch in MLP-to-DPT migration')
    actual = model.state_dict()
    if any(not torch.equal(v.cpu(), actual[k].cpu()) for k,v in shared.items()):
        raise ValueError('A shared tensor changed during MLP-to-DPT migration')
    return [k for k in source if k.startswith('surface_head.')]
