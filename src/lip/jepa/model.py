from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import FramePacket, MemoryState, CompletionOutput
from .positions import PositionEncoding
from .evidence import EvidenceHead
from .predictor import DecoderBlock
from .memory import SourceWriter, read_memory


class DinoJEPA(nn.Module):
    architecture_id = 'dino_jepa_v1'

    def __init__(self, encoder=None, dim=256, depth=4, memory_enabled=False,
                 geometry_enabled=False, weights_version='untrained', memory_policy='both'):
        super().__init__()
        self.encoder = encoder
        self.weights_version = weights_version
        self.memory_enabled = memory_enabled
        if memory_policy not in ('both', 'dynamic', 'persistent', 'none'):
            raise ValueError('Unknown memory read policy')
        self.memory_policy = memory_policy
        self.src_proj = nn.Linear(768, dim)
        self.positions = PositionEncoding(dim)
        self.evidence = EvidenceHead(dim)
        self.mask_query = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.blocks = nn.ModuleList([DecoderBlock(dim) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim)
        self.feature_last = nn.Linear(dim, 384)
        self.feature_mid = nn.Linear(dim, 384)
        self.support = nn.Linear(dim, 1)
        self.log_error = nn.Linear(dim, 1)
        self.writer = SourceWriter(dim)
        self.memory_types = nn.Embedding(2, dim)
        self.memory_age_rate = nn.Parameter(torch.tensor(-2.))
        self.geometry_adapter = None
        if geometry_enabled:
            from .geometry_adapter import GeometryAdapter
            self.geometry_adapter = GeometryAdapter(dim)

    def empty_state(self, packet):
        return MemoryState(packet.stream_id, self.weights_version)

    def readable_memory(self, state):
        if self.memory_policy == 'both':
            return state
        return replace(state, persistent=state.persistent if self.memory_policy == 'persistent' else (),
            dynamic=state.dynamic if self.memory_policy == 'dynamic' else ())

    def forward(self, packet: FramePacket, memory=None):
        if self.encoder is None:
            raise RuntimeError('RGB forward requires a real frozen encoder')
        return self.forward_features(packet, *self.encoder(packet.rgb_crop), memory=memory)

    def observe_features(self, packet, mid, last, memory):
        """Exact source-writer path for representation-only burn-in, without unused queries."""
        packet.validate()
        if memory.weights_version != self.weights_version or memory.stream_id != packet.stream_id:
            raise ValueError('Stale/cross-stream burn-in state')
        if memory.last_timestamp_s is not None and not torch.all(packet.timestamp_s > memory.last_timestamp_s):
            raise ValueError('Noncausal burn-in')
        h = self.src_proj(torch.cat((mid, last), -1))
        position, prompt, _, xy = self.positions(packet, memory.last_timestamp_s)
        valid = F.avg_pool2d(packet.pixel_valid.float(), 14, 14).flatten(1) >= .999
        mem, mv, _, kinds = read_memory(self.readable_memory(memory), packet)
        past = None
        if mem is not None:
            object_valid = mv & (kinds == 0)
            past = (mem * object_valid[..., None]).sum(1) / object_valid.sum(-1).clamp_min(1)[:, None]
        _, p, _, _, _ = self.evidence(h, position, prompt, valid, past)
        return self.writer(h, position, p, valid, xy, packet, memory)

    def forward_features(self, packet: FramePacket, mid, last, memory=None):
        """Exact-transform cached raw encoder features; never masked clean tokens."""
        packet.validate()
        memory = self.empty_state(packet) if memory is None else memory
        if memory.weights_version != self.weights_version or memory.stream_id != packet.stream_id:
            raise ValueError('Stale weights or cross-stream state')
        if memory.last_timestamp_s is not None and not torch.all(packet.timestamp_s > memory.last_timestamp_s):
            raise ValueError('Prediction cannot read its own or future committed source')
        h = self.src_proj(torch.cat((mid, last), -1))
        position, prompt, dt, xy = self.positions(packet, memory.last_timestamp_s)
        valid = F.avg_pool2d(packet.pixel_valid.float(), 14, 14).flatten(1) >= .999
        mem, mv, age, kinds = read_memory(self.readable_memory(memory), packet) if self.memory_enabled else (None,) * 4
        past = None
        if mem is not None:
            object_valid = mv & (kinds == 0)
            past = (mem * object_valid[..., None]).sum(1) / object_valid.sum(-1).clamp_min(1)[:, None]
        logits, p, current, cv, cb = self.evidence(h, position, prompt, valid, past)
        mb = None
        if mem is not None:
            mem = mem + self.memory_types(kinds)
            mb = -F.softplus(self.memory_age_rate) * age.clamp_min(0).log1p()
        geometry, gv, gq = None, None, None
        if packet.geometry is not None:
            if self.geometry_adapter is None:
                raise ValueError('Geometry supplied to a model without the adapter')
            geometry, gv, gq = self.geometry_adapter(packet.geometry)
        q = self.mask_query + position + prompt[:, None] + dt[:, None]
        for block in self.blocks:
            q = block(q, current, cv, cb, mem, mv, mb, geometry, gv, gq)
        q = self.final_norm(q)
        proposal = self.writer(h, position, p, valid, xy, packet, memory) if self.memory_enabled else None
        support = self.support(q).squeeze(-1)
        return CompletionOutput(q, last, self.feature_last(q), self.feature_mid(q), p,
            support.sigmoid(), self.log_error(q).squeeze(-1).clamp(-14, 5),
            dict(raw='current_pixel_encoded', predicted='latent_hypothesis', current_valid=valid,
                 memory_tokens=0 if mem is None else mem.shape[1], stream_id=packet.stream_id,
                 timestamp_s=packet.timestamp_s, weights_version=self.weights_version),
            proposal, logits, support)
