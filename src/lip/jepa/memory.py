"""Bounded source-only evidence records, committed after the current prediction."""
from dataclasses import replace
import torch
from torch import nn
from .contracts import SourceRecord, MemoryProposal
from .predictor import SafeAttention
from .evidence import pool_grid


class SourceWriter(nn.Module):
    def __init__(self, dim=256, reliable_score=.8, reliable_fraction=.25):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, 16, dim) * .02)
        self.attention = SafeAttention(dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.reliable_score = reliable_score
        self.reliable_fraction = reliable_fraction

    def forward(self, source, position, p_obs, valid, xy, packet, state):
        # This signature intentionally cannot consume generated latent/geometry/labels.
        b = len(source)
        fraction = ((p_obs.detach() >= self.reliable_score) & valid).sum(-1) / valid.sum(-1).clamp_min(1)
        write = (fraction >= self.reliable_fraction) & valid.any(-1)
        h = self.attention(self.queries.expand(b, -1, -1), source + position, valid, p_obs.clamp_min(.05).log())
        h = h + self.ffn(self.norm(h))
        mass = p_obs * valid
        centroid = (xy * mass[..., None]).sum(1) / mass.sum(1).clamp_min(1e-6)[:, None]
        coordinates = centroid[:, None].expand(-1, 16, -1)
        quality = (p_obs * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        object_valid = write[:, None].expand(-1, 16)
        common = dict(timestamp_s=packet.timestamp_s, stream_id=packet.stream_id,
                      weights_version=state.weights_version, quality=quality.detach(), affine=packet.A_image_to_crop)
        obj = SourceRecord(h, object_valid, source_xy=coordinates, kind=torch.zeros(b, 16, device=h.device, dtype=torch.long), **common)
        context, cv = pool_grid(source + position, valid, 4)
        # Adjacent pooled cells form eight context summaries without inventing object evidence.
        context = context.reshape(b, 8, 2, -1).mean(2)
        cv = cv.reshape(b, 8, 2).any(2)
        pooled_xy, xy_valid = pool_grid(xy, valid, 4)
        context_xy = pooled_xy.reshape(b, 8, 2, 2).sum(2) / xy_valid.reshape(b, 8, 2).sum(2).clamp_min(1)[..., None]
        dynamic = SourceRecord(torch.cat((h[:, :8], context), 1), torch.cat((object_valid[:, :8], cv), 1),
            source_xy=torch.cat((coordinates[:, :8], context_xy), 1),
            kind=torch.cat((torch.zeros(b, 8, device=h.device, dtype=torch.long), torch.ones(b, 8, device=h.device, dtype=torch.long)), 1), **common)
        return MemoryProposal(obj, dynamic, write, state.generation)


def _merge(old, new, choose):
    values = {}
    for name in ('features', 'valid', 'timestamp_s', 'source_xy', 'kind', 'quality', 'affine'):
        a, b = getattr(old, name), getattr(new, name)
        values[name] = torch.where(choose.reshape(-1, *([1] * (a.ndim - 1))), b, a)
    return replace(old, **values)


def commit(state, proposal, detach=False):
    if proposal.expected_generation != state.generation:
        raise ValueError('Duplicate or stale memory commit')
    record = proposal.object_record
    if record.stream_id != state.stream_id or record.weights_version != state.weights_version:
        raise ValueError('Memory identity/version mismatch')
    if state.last_timestamp_s is not None and not torch.all(record.timestamp_s > state.last_timestamp_s):
        raise ValueError('Memory must commit strictly increasing timestamps')
    slots = list(state.persistent)
    if len(slots) < 4:
        if slots:
            capture = proposal.object_write & ~slots[0].valid.any(-1)
            slots[0] = _merge(slots[0], record, capture)
            record = replace(record, valid=record.valid & ~capture[:, None])
        slots.append(record)
    else:
        # Keep the first reliable record in each lane. Empty lanes can acquire an anchor later.
        anchor_empty = ~slots[0].valid.any(-1)
        slots[0] = _merge(slots[0], record, proposal.object_write & anchor_empty)
        candidates = torch.stack([r.quality for r in slots[1:]], 1)
        cover = torch.stack([torch.nn.functional.cosine_similarity(
            r.features.detach().float().mean(1), record.features.detach().float().mean(1), dim=-1) for r in slots[1:]], 1)
        # Prefer replacing redundant, lower-quality records; invalid records always go first.
        valid = torch.stack([r.valid.any(-1) for r in slots[1:]], 1)
        score = torch.where(valid, candidates - .1 * cover, candidates.new_full(candidates.shape, -10.))
        index = score.argmin(1)
        for i in range(1, 4):
            slots[i] = _merge(slots[i], record, proposal.object_write & ~anchor_empty & (index == i - 1))
    new = replace(state, persistent=tuple(slots), dynamic=(state.dynamic + (proposal.dynamic_record,))[-4:],
                  last_timestamp_s=record.timestamp_s.clone(), generation=state.generation + 1)
    return new.detach() if detach else new


def read_memory(state, packet):
    if state.stream_id != packet.stream_id:
        raise ValueError('Cross-stream memory read')
    records = state.persistent + state.dynamic
    if not records:
        return None, None, None, None
    valid, ages = [], []
    for r in records:
        if r.weights_version != state.weights_version or r.stream_id != state.stream_id:
            raise ValueError('Stale/cross-stream source record')
        age = packet.timestamp_s - r.timestamp_s
        valid.append(r.valid & (age > 0)[:, None])
        ages.append(age.float()[:, None].expand_as(r.valid))
    return (torch.cat([r.features for r in records], 1), torch.cat(valid, 1),
            torch.cat(ages, 1), torch.cat([r.kind for r in records], 1))
