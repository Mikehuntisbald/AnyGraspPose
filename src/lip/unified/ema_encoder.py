"""Explicit online DINO adaptation and optimizer-step EMA target ownership."""
import copy
import math
import types
from dataclasses import fields, replace
import torch
from torch.nn import functional as F


class _PositionResize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, patches, matrix, kwargs):
        ctx.save_for_backward(matrix)
        ctx.source_shape = patches.shape
        return F.interpolate(patches, mode='bicubic', **kwargs)

    @staticmethod
    def backward(ctx, grad):
        matrix, = ctx.saved_tensors
        # Interpolation is linear. Its fixed transpose has no atomic scatter.
        with torch.autocast(grad.device.type, enabled=False):
            result = grad.float().flatten(2) @ matrix.float()
        return result.reshape(ctx.source_shape), None, None


def deterministic_position(self, x, w, h):
    n = self.pos_embed.shape[1]-1
    if x.shape[1]-1 == n and w == h:
        return self.pos_embed
    side = math.isqrt(n)
    shape = (w//self.patch_size, h//self.patch_size)
    kwargs = dict(antialias=self.interpolate_antialias)
    if self.interpolate_offset:
        kwargs['scale_factor'] = tuple((s+self.interpolate_offset)/side for s in shape)
    else:
        kwargs['size'] = shape
    key = (shape, str(x.device))
    if key not in self._ema_position_matrices:
        with torch.no_grad():
            basis = torch.eye(n, device=x.device).reshape(n, 1, side, side)
            matrix = F.interpolate(basis, mode='bicubic', **kwargs).flatten(1).T.contiguous()
            self._ema_position_matrices[key] = matrix
    pos = self.pos_embed.float()
    patches = pos[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
    patches = _PositionResize.apply(patches, self._ema_position_matrices[key], kwargs)
    patches = patches.flatten(2).transpose(1, 2)
    return torch.cat((pos[:, :1], patches), dim=1).to(x.dtype)


def enable_deterministic_position(backbone):
    backbone._ema_position_matrices = {}
    backbone.interpolate_pos_encoding = types.MethodType(deterministic_position, backbone)


def attach_ema(model, config):
    plan = config['ema_encoder']
    if config['runtime'].get('reuse_teacher_real', True):
        raise ValueError('EMA targets must never reuse online student features')
    model.trainable_encoder = True
    model.model_version = 'local-cad-online-dino-ema-v13'
    model.encoder.trainable_encoder = True
    model.encoder.requires_grad_(True)
    model.ema_teacher = copy.deepcopy(model.encoder)
    if config.get("dino_layers"):
        model.ema_teacher.set_feature_layers(config["dino_layers"]["teacher"])
        model.model_version="local-cad-dino4-11-equal-v14"
    if config.get('local_structure'):
        model.model_version='local-cad-multiscale-difference-v15'
    model.ema_teacher.trainable_encoder = False
    model.ema_teacher.requires_grad_(False).eval()
    if hasattr(model.encoder, 'backbone'):
        enable_deterministic_position(model.encoder.backbone)
    model.ema_momentum = float(plan['momentum'])
    if not 0 < model.ema_momentum < 1:
        raise ValueError('EMA momentum must be strictly between zero and one')
    model.register_buffer('ema_updates', torch.zeros((), dtype=torch.long,
                          device=next(model.parameters()).device))


@torch.no_grad()
def update_ema(model):
    """Call exactly once after a successful synchronized optimizer update."""
    source = dict(model.encoder.named_parameters())
    for name, target in model.ema_teacher.named_parameters():
        target.mul_(model.ema_momentum).add_(source[name], alpha=1-model.ema_momentum)
    source_buffers = dict(model.encoder.named_buffers())
    for name, target in model.ema_teacher.named_buffers():
        target.copy_(source_buffers[name])
    model.ema_updates.add_(1)


def detach_observation(obs):
    return replace(obs, **{f.name: getattr(obs, f.name).detach()
                         for f in fields(obs) if isinstance(getattr(obs, f.name), torch.Tensor)})
