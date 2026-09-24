"""Deployment observations and training labels deliberately have separate types."""
from dataclasses import dataclass, field, replace
from typing import Optional
import torch


@dataclass(frozen=True)
class GeometryPacket:
    channels: torch.Tensor
    available: torch.Tensor
    quality: torch.Tensor


@dataclass(frozen=True)
class FramePacket:
    rgb_crop: torch.Tensor
    pixel_valid: torch.Tensor
    A_image_to_crop: torch.Tensor
    timestamp_s: torch.Tensor
    stream_id: tuple[str, ...]
    track_prompt: torch.Tensor  # Bx4 crop bbox, obtained from an allowed initializer
    source_size_wh: torch.Tensor
    K_image: Optional[torch.Tensor] = None
    geometry: Optional[GeometryPacket] = None

    def validate(self):
        b = len(self.rgb_crop)
        if self.rgb_crop.shape != (b, 3, 224, 224):
            raise ValueError('Expected normalized RGB Bx3x224x224')
        if self.pixel_valid.shape != (b, 1, 224, 224) or self.pixel_valid.dtype != torch.bool:
            raise ValueError('pixel_valid is an image-boundary bool mask')
        if self.A_image_to_crop.shape != (b, 3, 3) or self.track_prompt.shape != (b, 4):
            raise ValueError('Invalid crop affine or target prompt')
        if self.timestamp_s.shape != (b,) or self.timestamp_s.dtype != torch.float64:
            raise ValueError('Timestamps must be float64 seconds')
        if len(self.stream_id) != b or len(set(self.stream_id)) != b:
            raise ValueError('Each lane must have an independent stream identity')


@dataclass(frozen=True)
class TrainTargets:
    teacher_features_mid: torch.Tensor
    teacher_features_last: torch.Tensor
    feature_target_valid: torch.Tensor
    added_occlusion_fraction: torch.Tensor
    object_visible_target: Optional[torch.Tensor] = None
    object_support_target: Optional[torch.Tensor] = None
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SourceRecord:
    features: torch.Tensor
    valid: torch.Tensor
    timestamp_s: torch.Tensor
    source_xy: torch.Tensor
    kind: torch.Tensor  # 0 object, 1 context; context is never object evidence
    quality: torch.Tensor
    stream_id: tuple[str, ...]
    weights_version: str
    affine: torch.Tensor

    def detach(self):
        return replace(self, **{n: getattr(self, n).detach() for n in
            ('features', 'valid', 'timestamp_s', 'source_xy', 'kind', 'quality', 'affine')})


@dataclass(frozen=True)
class MemoryState:
    stream_id: tuple[str, ...]
    weights_version: str
    persistent: tuple[SourceRecord, ...] = ()
    dynamic: tuple[SourceRecord, ...] = ()
    last_timestamp_s: Optional[torch.Tensor] = None
    generation: int = 0

    def detach(self):
        return replace(self, persistent=tuple(r.detach() for r in self.persistent),
                       dynamic=tuple(r.detach() for r in self.dynamic))


@dataclass(frozen=True)
class MemoryProposal:
    object_record: SourceRecord
    dynamic_record: SourceRecord
    object_write: torch.Tensor
    expected_generation: int


@dataclass
class CompletionOutput:
    z_state: torch.Tensor
    f_observed: torch.Tensor
    f_predicted: torch.Tensor
    f_mid_predicted: torch.Tensor
    observation_evidence: torch.Tensor
    amodal_support: torch.Tensor
    log_feature_error: torch.Tensor
    source_metadata: dict
    next_memory_proposal: Optional[MemoryProposal]
    evidence_logits: torch.Tensor
    support_logits: torch.Tensor

    @property
    def f_fused(self):
        p = self.observation_evidence[..., None]
        return p * self.f_observed + (1 - p) * self.f_predicted
