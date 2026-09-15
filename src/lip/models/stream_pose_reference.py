"""Spatial keyframes plus a separate, immutable pose-reference state."""
from dataclasses import dataclass, replace
import torch
from lip.engine.stream_state import TemporalCache, CrossCache
from lip.models.stream_spatial_memory import SpatialRKTracker, SpatialCache
from lip.models.pose_reference import PoseReferenceFeedback, reference_action, apply_reference_feedback


@dataclass(frozen=True)
class PoseReference:
    pose: torch.Tensor
    started_timestamp: torch.Tensor  # Residence clock: first update or external correction, not measurement time.
    stream_tag: torch.Tensor

    def detach(self):
        return PoseReference(self.pose.detach(), self.started_timestamp.detach(), self.stream_tag.detach())

    @classmethod
    def capture(cls, pose, timestamp, tag):
        return cls(pose.detach().float().clone(), timestamp.detach().clone(), tag.detach().clone())


@dataclass(frozen=True)
class PoseReferenceCache(SpatialCache):
    reference: PoseReference | None = None

    def detach(self):
        spatial = super().detach()
        return PoseReferenceCache(**spatial.__dict__, reference=None if self.reference is None else self.reference.detach())

    def clear_visual_history(self):
        # The feature-history ablation keeps nonvisual state (pose and motion).
        return PoseReferenceCache(TemporalCache(capacity=self.capacity), variant=self.variant, reference=self.reference)

    @property
    def kv_bytes(self):
        extra = 0 if self.reference is None else sum(t.numel() * t.element_size() for t in
            (self.reference.pose, self.reference.started_timestamp, self.reference.stream_tag))
        return super().kv_bytes + extra


class PoseReferenceRKTracker(SpatialRKTracker):
    def __init__(self, memory_frames=8, dropout=0., time_unit=1/30, max_gap_seconds=.5,
                 slots=4, min_gap=4, max_age=64, tolerance=.05, dense_side=14):
        super().__init__(memory_frames, dropout, time_unit, max_gap_seconds, slots, min_gap, max_age, tolerance, dense_side)
        self.architecture_id = 'stream_rk_pose_reference'
        self.cache_contract = 'lip-rk-spatial-pose-reference-v1'
        self.variant = f'R1K1_spatial{dense_side}_pose_reference'
        self.pose_reference_feedback = PoseReferenceFeedback()

    def forward(self, features, metadata, cache=None, profiler=None):
        if cache is not None and not isinstance(cache, PoseReferenceCache):
            if not isinstance(cache, CrossCache) or isinstance(cache, SpatialCache) or cache.metadata:
                raise ValueError('Pose-reference architecture requires its own cache')
        old_reference = getattr(cache, 'reference', None)
        base = features['T_base_centered']
        fresh = PoseReference.capture(base, metadata.timestamp, metadata.stream_tag)
        if old_reference is not None:
            if old_reference.pose.shape != base.shape:
                raise ValueError('Pose-reference batch changed; reset the stream')
            keep = (old_reference.stream_tag == metadata.stream_tag) & (old_reference.started_timestamp <= metadata.timestamp)
            reference = PoseReference(torch.where(keep[:, None, None], old_reference.pose, fresh.pose),
                torch.where(keep, old_reference.started_timestamp, fresh.started_timestamp), fresh.stream_tag)
        else:
            reference = fresh
        parent, spatial = super().forward(features, metadata, cache, profiler)
        target = reference_action(base, reference.pose, features['object_diameter_m'])
        action = torch.cat((parent['delta_rotvec'].float(), parent['delta_center_norm'].float()), -1)
        age = ((metadata.timestamp - reference.started_timestamp) / self.temporal.time_unit).clamp_min(0).float()
        coefficient = self.pose_reference_feedback(parent['latent'], features['state_input'], target - action,
            parent['observation_support'], age)
        result = apply_reference_feedback(parent, features, target, coefficient)
        result['reference_age_frames'] = age
        return result, PoseReferenceCache(**spatial.__dict__, reference=reference)

    def correct(self, state, pose_original, timestamp=None, relocalization=False):
        corrected = super().correct(state, pose_original, timestamp, relocalization)
        if relocalization or not isinstance(state.cache, PoseReferenceCache):
            return corrected
        # External correction refreshes this nonvisual prior only. Accepted LIP
        # proposals never refresh it; visual KV and keyframes remain untouched.
        old = state.cache.reference
        if old is None:
            return corrected
        reference = PoseReference.capture(corrected.pose_centered[None],
            old.started_timestamp.new_full((1,), corrected.timestamp), old.stream_tag)
        return replace(corrected, cache=replace(state.cache, reference=reference))
