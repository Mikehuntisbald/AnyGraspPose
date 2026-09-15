"""Adaptive pose memory with a separately retained initial rotation for reading."""
from dataclasses import dataclass,replace
import torch
from lip.engine.stream_state import TemporalCache,CrossCache
from lip.models.stream_spatial_memory import SpatialRKTracker,SpatialCache
from lip.models.stream_adaptive_reference import AdaptiveReference,AdaptiveReferenceCache,AdaptiveReferenceRKTracker
from lip.models.pose_reference import reference_action,apply_reference_feedback
from lip.models.adaptive_reference import write_innovation,write_reference
from lip.models.rotation_anchor import RotationAnchorReadout,angular_evidence,blend_rotation_reference


@dataclass(frozen=True)
class RotationAnchorCache(AdaptiveReferenceCache):
    initial_rotation:torch.Tensor|None=None

    def detach(self):
        adaptive=super().detach()
        return RotationAnchorCache(**adaptive.__dict__,initial_rotation=None if self.initial_rotation is None else self.initial_rotation.detach())

    def clear_visual_history(self):
        return RotationAnchorCache(TemporalCache(capacity=self.capacity),variant=self.variant,
            reference=self.reference,initial_rotation=self.initial_rotation)

    @property
    def kv_bytes(self):
        extra=0 if self.initial_rotation is None else self.initial_rotation.numel()*self.initial_rotation.element_size()
        return super().kv_bytes+extra


class RotationAnchorRKTracker(AdaptiveReferenceRKTracker):
    def __init__(self,memory_frames=8,dropout=0.,time_unit=1/30,max_gap_seconds=.5,
                 slots=4,min_gap=4,max_age=64,tolerance=.05,dense_side=14,write_limit=.25):
        super().__init__(memory_frames,dropout,time_unit,max_gap_seconds,slots,min_gap,max_age,tolerance,dense_side,write_limit)
        self.architecture_id='stream_rk_rotation_anchor'
        self.cache_contract='lip-rk-rotation-anchor-v1'
        self.variant=f'R1K1_spatial{dense_side}_rotation_anchor'
        self.rotation_anchor_readout=RotationAnchorReadout()

    def forward(self,features,metadata,cache=None,profiler=None):
        if cache is not None and not isinstance(cache,RotationAnchorCache):
            if not isinstance(cache,CrossCache) or isinstance(cache,SpatialCache) or cache.metadata:
                raise ValueError('Rotation-anchor architecture requires its own cache')
        base=features['T_base_centered'];fresh=AdaptiveReference.capture(base,metadata.timestamp,metadata.stream_tag)
        initial_rotation=base[:,:3,:3].detach().float().clone()
        old=getattr(cache,'reference',None)
        if old is not None:
            if old.pose.shape!=base.shape or cache.initial_rotation is None or cache.initial_rotation.shape!=initial_rotation.shape:
                raise ValueError('Rotation-anchor batch changed or anchor missing; reset stream')
            keep=(old.stream_tag==metadata.stream_tag)&(old.started_timestamp<=metadata.timestamp)&(old.observed_timestamp<metadata.timestamp)
            reference=AdaptiveReference(torch.where(keep[:,None,None],old.pose,fresh.pose),
                torch.where(keep,old.started_timestamp,fresh.started_timestamp),fresh.stream_tag,
                torch.where(keep,old.observed_timestamp,fresh.observed_timestamp))
            initial_rotation=torch.where(keep[:,None,None],cache.initial_rotation.detach(),initial_rotation)
        else:reference=fresh
        parent,spatial=SpatialRKTracker.forward(self,features,metadata,cache,profiler)
        target=reference_action(base,reference.pose,features['object_diameter_m'],detach_reference=False)
        action=torch.cat((parent['delta_rotvec'].float(),parent['delta_center_norm'].float()),-1)
        age=((metadata.timestamp-reference.started_timestamp)/self.temporal.time_unit).clamp_min(0).float()
        # Keep the parent's coefficient inputs and center read exactly intact
        # for identical features/reference. Only the rotation target changes.
        coefficients=self.pose_reference_feedback(parent['latent'],features['state_input'],target-action,parent['observation_support'],age)
        evidence=angular_evidence(reference.pose,initial_rotation,parent['pose_centered'])
        fraction=self.rotation_anchor_readout(parent['latent'],features['state_input'],evidence,parent['observation_support'],age)
        effective=blend_rotation_reference(reference.pose,initial_rotation,fraction)
        effective_target=reference_action(base,effective,features['object_diameter_m'],detach_reference=False)
        mixed_target=torch.cat((effective_target[:,:3],target[:,3:]),-1)
        result=apply_reference_feedback(parent,features,mixed_target,coefficients)
        result.update(reference_age_frames=age,rotation_anchor_fraction=fraction,
            rotation_anchor_gap_norm=evidence[:,:3].norm(dim=-1))
        # Continue writing both adaptive channels from the pre-feedback visual
        # proposal. The immutable rotation is never overwritten by actor output.
        innovation,valid=write_innovation(reference.pose,parent['pose_centered'],features['object_diameter_m'])
        gates=self.reference_writer(parent['latent'],features['state_input'],innovation,parent['observation_support'],age)
        gates=torch.where(valid[:,None],gates,torch.zeros_like(gates))
        pose,rotation_norm,center_norm=write_reference(reference.pose,innovation,gates,features['object_diameter_m'])
        updated=AdaptiveReference(pose,reference.started_timestamp,reference.stream_tag,metadata.timestamp.detach().clone())
        result.update(reference_write_rotation_coefficient=gates[:,0],reference_write_center_coefficient=gates[:,1],
            reference_write_rotation_norm=rotation_norm,reference_write_center_norm=center_norm,reference_write_valid_target=valid)
        return result,RotationAnchorCache(**spatial.__dict__,reference=updated,initial_rotation=initial_rotation)

    def correct(self,state,pose_original,timestamp=None,relocalization=False):
        corrected=SpatialRKTracker.correct(self,state,pose_original,timestamp,relocalization)
        if relocalization or not isinstance(state.cache,RotationAnchorCache) or state.cache.reference is None:return corrected
        old=state.cache.reference
        reference=AdaptiveReference.capture(corrected.pose_centered[None],old.started_timestamp.new_full((1,),corrected.timestamp),old.stream_tag)
        return replace(corrected,cache=replace(state.cache,reference=reference,
            initial_rotation=corrected.pose_centered[None,:3,:3].detach().float().clone()))
