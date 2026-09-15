"""Immutable streaming state. Cache tensors are never edited after arrival."""
from dataclasses import dataclass, replace
from typing import Optional
import torch

CACHE_CONTRACT = 'lip-v2-source17-block-local-timebias-v1'
CROSS_CACHE_CONTRACT = 'lip-v2-source17-context-readout-cross-gated-v1'


def cache_contract_for(architecture):
    if architecture=='stream_rk_rotation_anchor_smooth':return 'lip-rk-rotation-anchor-smooth-v1'
    if architecture=='stream_rk_rotation_anchor':return 'lip-rk-rotation-anchor-v1'
    if architecture=='stream_rk_adaptive_reference':return 'lip-rk-adaptive-reference-v1'
    if architecture=='stream_rk_pose_reference':return 'lip-rk-spatial-pose-reference-v1'
    if architecture=='stream_rk_direct_pose':return 'lip-rk-spatial-direct-pose-v1'
    if architecture=='stream_rk_aligned':return 'lip-rk-aligned-memory-v1'
    if architecture=='stream_rk_spatial':return 'lip-rk-spatial-memory-v1'
    if architecture=='stream_rk_factorial':return 'lip-rk-factorial-v1'
    if architecture=='stream_dual_cross_residual':return 'lip-v2-source17-context-cross-parent-residual-v1'
    return CROSS_CACHE_CONTRACT if architecture=='stream_dual_cross' else CACHE_CONTRACT


@dataclass(frozen=True)
class FrameMeta:
    timestamp: torch.Tensor       # [B], float64 seconds, never BF16 epoch seconds
    frame_id: torch.Tensor        # [B], int64
    stream_tag: torch.Tensor      # [B], int64 reset generation / stream identity
    key_valid: torch.Tensor       # [B,17], True = allowed source key
    role_bias: torch.Tensor       # [B,17], immutable object-readout key bias

    def __post_init__(self):
        if self.frame_id.ndim!=1 or self.timestamp.shape!=self.frame_id.shape or self.stream_tag.shape!=self.frame_id.shape:
            raise ValueError('Timestamp, frame ID and stream tag must each have shape [B]')
        b = self.frame_id.shape[0]
        if self.timestamp.dtype != torch.float64 or self.frame_id.dtype != torch.int64:
            raise TypeError('Use float64 timestamps and int64 frame IDs')
        if self.stream_tag.dtype != torch.int64 or self.key_valid.dtype != torch.bool:
            raise TypeError('Invalid source identity or valid-mask dtype')
        if self.key_valid.shape != (b, 17) or self.role_bias.shape != (b, 17):
            raise ValueError('Each frame has exactly 17 persistent source tokens')

    def detach(self):
        return FrameMeta(*(getattr(self, f).detach().clone() for f in self.__dataclass_fields__))


@dataclass(frozen=True)
class KVBlock:
    key: torch.Tensor            # [B,8,17,32], pre-norm projected K actually consumed
    value: torch.Tensor

    def detach(self):
        return KVBlock(self.key.detach(), self.value.detach())


@dataclass(frozen=True)
class TemporalCache:
    layers: tuple[tuple[KVBlock, ...], ...] = ((), (), (), ())
    metadata: tuple[FrameMeta, ...] = ()
    capacity: int = 8
    contract: str = CACHE_CONTRACT

    def __post_init__(self):
        if self.capacity < 1 or len(self.metadata) > self.capacity:
            raise ValueError('Invalid cache capacity')
        if len(self.layers) != 4 or any(len(layer) != len(self.metadata) for layer in self.layers):
            raise ValueError('Each layer must retain the same frame blocks')
        if self.contract != CACHE_CONTRACT:
            raise ValueError('Incompatible cache contract')

    def detach(self):
        return TemporalCache(tuple(tuple(b.detach() for b in layer) for layer in self.layers),
                             tuple(m.detach() for m in self.metadata), self.capacity)

    @property
    def kv_bytes(self):
        return sum(t.numel()*t.element_size() for layer in self.layers for b in layer for t in (b.key,b.value))


@dataclass(frozen=True)
class ContextBlock:
    key: torch.Tensor             # [B,8,1,32], context readout only
    value: torch.Tensor
    metadata: FrameMeta

    def detach(self):
        return ContextBlock(self.key.detach(),self.value.detach(),self.metadata.detach())


@dataclass(frozen=True)
class CrossCache:
    """Separate source and context KV; updated object readout enters neither."""
    source: TemporalCache
    contexts: tuple[ContextBlock,...] = ()
    contract: str = CROSS_CACHE_CONTRACT

    def __post_init__(self):
        if self.contract!=CROSS_CACHE_CONTRACT or len(self.contexts)!=len(self.source.metadata):
            raise ValueError('Source and context cache must have aligned frame blocks')

    @property
    def layers(self):return self.source.layers

    @property
    def metadata(self):return self.source.metadata

    @property
    def capacity(self):return self.source.capacity

    @property
    def kv_bytes(self):
        return self.source.kv_bytes+sum(t.numel()*t.element_size() for b in self.contexts for t in (b.key,b.value))

    def detach(self):
        return CrossCache(self.source.detach(),tuple(b.detach() for b in self.contexts))


@dataclass(frozen=True)
class RingCache:
    """Persistent ring of tensor references; bounded storage, no in-place autograd writes.

    This does not claim to be a preallocated CUDA tensor optimization. Physical
    slot order is independent of frame IDs; attention still validates identities.
    """
    slots: tuple
    cursor: int = 0
    count: int = 0

    @classmethod
    def empty(cls, capacity):
        return cls((None,)*capacity)

    def append(self, blocks, metadata):
        slots=list(self.slots); slots[self.cursor]=(tuple(blocks),metadata)
        return RingCache(tuple(slots),(self.cursor+1)%len(slots),min(self.count+1,len(slots)))

    def functional(self):
        indices=[(self.cursor-self.count+i)%len(self.slots) for i in range(self.count)]
        frames=[self.slots[i] for i in indices]
        return TemporalCache(tuple(tuple(f[0][l] for f in frames) for l in range(4)),
                             tuple(f[1] for f in frames),len(self.slots))

    @property
    def kv_bytes(self):
        return self.functional().kv_bytes


@dataclass(frozen=True)
class SourceGeometry:
    timestamp: float
    frame_id: int
    stream_id: str
    object_id: str
    camera_id: str
    A_source: torch.Tensor
    K_crop_source: torch.Tensor
    T_base_source: torch.Tensor
    original_image_shape: tuple[int,int]
    object_diameter: torch.Tensor
    mesh_hash: str
    key_valid: torch.Tensor
    silhouette_token_fraction: torch.Tensor


@dataclass(frozen=True)
class StreamState:
    cache: TemporalCache | RingCache | CrossCache
    pose_centered: torch.Tensor
    timestamp: float
    previous_pose: Optional[torch.Tensor]
    previous_timestamp: Optional[float]
    frame_id: int
    stream_id: str
    object_id: str
    camera_id: str
    mesh_hash: str
    mesh: dict
    K: torch.Tensor
    image_shape: tuple[int,int]
    weights_version: int
    parameter_versions: tuple
    source_metadata: tuple[SourceGeometry,...] = ()
    pending_pose: Optional[torch.Tensor] = None
    pending_timestamp: Optional[float] = None
    generation: int = 0
    cache_contract: str = CACHE_CONTRACT

    def detach(self):
        if isinstance(self.cache,RingCache):
            cache=self.cache.functional().detach()
        else:cache=self.cache.detach()
        return replace(self,cache=cache,pose_centered=self.pose_centered.detach(),
                       previous_pose=None if self.previous_pose is None else self.previous_pose.detach())
