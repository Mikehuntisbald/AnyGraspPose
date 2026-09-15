"""Spatial LIP with a reference written only after its current pose is read."""
from dataclasses import dataclass,replace
import torch
from lip.engine.stream_state import TemporalCache,CrossCache
from lip.models.stream_spatial_memory import SpatialRKTracker,SpatialCache
from lip.models.stream_pose_reference import PoseReferenceRKTracker
from lip.models.pose_reference import reference_action,apply_reference_feedback
from lip.models.adaptive_reference import ReferenceWriter,write_innovation,write_reference


@dataclass(frozen=True)
class AdaptiveReference:
    pose:torch.Tensor
    started_timestamp:torch.Tensor  # Residence since capture; writing does not reset this clock.
    stream_tag:torch.Tensor
    observed_timestamp:torch.Tensor  # Latest observation used by writer, even for a zero gate.

    def detach(self):
        return AdaptiveReference(*(getattr(self,n).detach() for n in self.__dataclass_fields__))

    @classmethod
    def capture(cls,pose,timestamp,tag):
        return cls(pose.detach().float().clone(),timestamp.detach().clone(),tag.detach().clone(),timestamp.detach().clone())


@dataclass(frozen=True)
class AdaptiveReferenceCache(SpatialCache):
    reference:AdaptiveReference|None=None

    def detach(self):
        spatial=super().detach()
        return AdaptiveReferenceCache(**spatial.__dict__,reference=None if self.reference is None else self.reference.detach())

    def clear_visual_history(self):
        return AdaptiveReferenceCache(TemporalCache(capacity=self.capacity),variant=self.variant,reference=self.reference)

    @property
    def kv_bytes(self):
        extra=0 if self.reference is None else sum(getattr(self.reference,n).numel()*getattr(self.reference,n).element_size() for n in self.reference.__dataclass_fields__)
        return super().kv_bytes+extra


class AdaptiveReferenceRKTracker(PoseReferenceRKTracker):
    def __init__(self,memory_frames=8,dropout=0.,time_unit=1/30,max_gap_seconds=.5,
                 slots=4,min_gap=4,max_age=64,tolerance=.05,dense_side=14,write_limit=.25):
        super().__init__(memory_frames,dropout,time_unit,max_gap_seconds,slots,min_gap,max_age,tolerance,dense_side)
        self.architecture_id='stream_rk_adaptive_reference'
        self.cache_contract='lip-rk-adaptive-reference-v1'
        self.variant=f'R1K1_spatial{dense_side}_adaptive_reference'
        self.reference_writer=ReferenceWriter(write_limit)

    def forward(self,features,metadata,cache=None,profiler=None):
        if cache is not None and not isinstance(cache,AdaptiveReferenceCache):
            if not isinstance(cache,CrossCache) or isinstance(cache,SpatialCache) or cache.metadata:
                raise ValueError('Adaptive-reference architecture requires its own cache')
        base=features['T_base_centered'];fresh=AdaptiveReference.capture(base,metadata.timestamp,metadata.stream_tag)
        old=getattr(cache,'reference',None)
        if old is not None:
            if old.pose.shape!=base.shape:raise ValueError('Adaptive-reference batch changed; reset stream')
            keep=(old.stream_tag==metadata.stream_tag)&(old.started_timestamp<=metadata.timestamp)&(old.observed_timestamp<metadata.timestamp)
            reference=AdaptiveReference(torch.where(keep[:,None,None],old.pose,fresh.pose),
                torch.where(keep,old.started_timestamp,fresh.started_timestamp),fresh.stream_tag,
                torch.where(keep,old.observed_timestamp,fresh.observed_timestamp))
        else:reference=fresh
        # Invoke the same spatial visual actor that the frozen parent uses.
        parent,spatial=SpatialRKTracker.forward(self,features,metadata,cache,profiler)
        target=reference_action(base,reference.pose,features['object_diameter_m'],detach_reference=False)
        action=torch.cat((parent['delta_rotvec'].float(),parent['delta_center_norm'].float()),-1)
        age=((metadata.timestamp-reference.started_timestamp)/self.temporal.time_unit).clamp_min(0).float()
        coefficients=self.pose_reference_feedback(parent['latent'],features['state_input'],target-action,parent['observation_support'],age)
        result=apply_reference_feedback(parent,features,target,coefficients)
        result['reference_age_frames']=age
        # Current output uses only the old reference. Store a detached proposal
        # from BEFORE reference feedback for later observations, with no GT.
        innovation,valid=write_innovation(reference.pose,parent['pose_centered'],features['object_diameter_m'])
        gates=self.reference_writer(parent['latent'],features['state_input'],innovation,parent['observation_support'],age)
        gates=torch.where(valid[:,None],gates,torch.zeros_like(gates))
        pose,rotation_norm,center_norm=write_reference(reference.pose,innovation,gates,features['object_diameter_m'])
        updated=AdaptiveReference(pose,reference.started_timestamp,reference.stream_tag,metadata.timestamp.detach().clone())
        result.update(reference_write_rotation_coefficient=gates[:,0],reference_write_center_coefficient=gates[:,1],
            reference_write_rotation_norm=rotation_norm,reference_write_center_norm=center_norm,reference_write_valid_target=valid)
        return result,AdaptiveReferenceCache(**spatial.__dict__,reference=updated)

    def correct(self,state,pose_original,timestamp=None,relocalization=False):
        corrected=SpatialRKTracker.correct(self,state,pose_original,timestamp,relocalization)
        if relocalization or not isinstance(state.cache,AdaptiveReferenceCache) or state.cache.reference is None:return corrected
        old=state.cache.reference
        reference=AdaptiveReference.capture(corrected.pose_centered[None],old.started_timestamp.new_full((1,),corrected.timestamp),old.stream_tag)
        return replace(corrected,cache=replace(state.cache,reference=reference))
