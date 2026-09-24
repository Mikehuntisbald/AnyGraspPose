import torch
from torch import nn
from torch.nn import functional as F


def pool_grid(h, valid, side):
    b, _, d = h.shape
    m = valid.reshape(b, 1, 16, 16).float()
    den = F.adaptive_avg_pool2d(m, side)
    values = F.adaptive_avg_pool2d(h.transpose(1, 2).reshape(b, d, 16, 16) * m, side) / den.clamp_min(1e-6)
    return values.flatten(2).transpose(1, 2), den.flatten(1) > 0


class EvidenceHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.types = nn.Parameter(torch.randn(2, dim) * .02)
        self.context_gate = nn.Parameter(torch.tensor(-1.))

    def forward(self, h, position, prompt, valid, past_summary=None):
        if past_summary is None:
            past_summary = torch.zeros_like(prompt)
        logits = self.head(torch.cat((h, prompt[:, None].expand_as(h), past_summary[:, None].expand_as(h)), -1)).squeeze(-1)
        p = logits.sigmoid() * valid
        context, context_valid = pool_grid(h + position, valid, 4)
        current = torch.cat((h + position + self.types[0],
                             self.context_gate.sigmoid() * context + self.types[1]), 1)
        bias = torch.cat((p.clamp_min(.05).log(), p.new_zeros(len(p), 16)), 1)
        return logits, p, current, torch.cat((valid, context_valid), 1), bias
