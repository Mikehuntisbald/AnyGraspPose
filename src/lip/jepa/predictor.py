import torch
from torch import nn
from torch.nn import functional as F


class SafeAttention(nn.Module):
    """SDPA with an explicitly non-evidence null source for empty lanes."""
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, query, source, valid=None, bias=None):
        b, n, d = query.shape
        if source is None or source.shape[1] == 0:
            return query * 0
        if valid is None:
            valid = torch.ones(source.shape[:2], device=source.device, dtype=torch.bool)
        nonempty = valid.any(-1)
        source = torch.cat((source, source.new_zeros(b, 1, d)), 1)
        valid = torch.cat((valid, ~nonempty[:, None]), 1)
        logits_bias = source.new_zeros(b, source.shape[1])
        if bias is not None:
            logits_bias[:, :-1] = bias.to(source.dtype)
        logits_bias = logits_bias.masked_fill(~valid, float('-inf'))
        q = self.q(query).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(source).reshape(b, -1, 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        # PyTorch 2.8 efficient SDPA omits aligned LSE when only its bias
        # requires gradients. Frozen warmup can create precisely that case:
        # history trains the evidence bias while current Q/K/V are constants.
        # A temporary Q leaf requests the normal backward buffers; it does not
        # unfreeze encoder/predictor weights or cut the history-bias gradient.
        if torch.is_grad_enabled() and logits_bias.requires_grad and not any(x.requires_grad for x in (q,k,v)):
            q.requires_grad_(True)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=logits_bias[:, None, None].to(q.dtype),
                                             dropout_p=0., is_causal=False)
        return self.out(out.transpose(1, 2).reshape(b, n, d)) * nonempty[:, None, None]


class DecoderBlock(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(5)])
        self.obs = SafeAttention(dim, heads)
        self.memory = SafeAttention(dim, heads)
        self.geometry = SafeAttention(dim, heads)
        self.self_attention = SafeAttention(dim, heads)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.geometry_gate = nn.Parameter(torch.tensor(-2.))

    def forward(self, q, current, current_valid, current_bias, memory=None, memory_valid=None,
                memory_bias=None, geometry=None, geometry_valid=None, geometry_quality=None):
        q = q + self.obs(self.norms[0](q), current, current_valid, current_bias)
        q = q + self.memory(self.norms[1](q), memory, memory_valid, memory_bias)
        if geometry is not None:
            q = q + self.geometry_gate.sigmoid() * geometry_quality[:, None, None] * self.geometry(
                self.norms[2](q), geometry, geometry_valid)
        h = self.norms[3](q)
        q = q + self.self_attention(h, h)
        return q + self.ffn(self.norms[4](q))
