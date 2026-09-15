"""Retain a trained spatial tracker and add a causal CAD-aligned local residual."""
from dataclasses import dataclass
import torch
from lip.models.stream_spatial_memory import SpatialRKTracker,SpatialCache,select_patches
from lip.models.stream_rk import AnchorBank
from lip.models.aligned_memory import AlignedMemoryReadout,pool_memory_geometry
from lip.engine.stream_state import CrossCache
from lip.geometry.so3 import update,original_pose


@dataclass(frozen=True)
class AlignedCache(SpatialCache):
    patch_geometry: torch.Tensor|None=None

    def detach(self):
        return AlignedCache(self.source.detach(),tuple(x.detach() for x in self.contexts),
            anchors=None if self.anchors is None else self.anchors.detach(),variant=self.variant,
            patches=None if self.patches is None else self.patches.detach(),patch_quality=None if self.patch_quality is None else self.patch_quality.detach(),
            patch_geometry=None if self.patch_geometry is None else self.patch_geometry.detach())

    @property
    def kv_bytes(self):
        return super().kv_bytes+(0 if self.patch_geometry is None else self.patch_geometry.numel()*self.patch_geometry.element_size())


class AlignedRKTracker(SpatialRKTracker):
    def __init__(self,memory_frames=8,dropout=0.,time_unit=1/30,max_gap_seconds=.5,slots=4,min_gap=4,max_age=64,tolerance=.05,dense_side=14,query_side=14,sigma=.05):
        super().__init__(memory_frames,dropout,time_unit,max_gap_seconds,slots,min_gap,max_age,tolerance,dense_side)
        if query_side not in (4,dense_side):raise ValueError('Query grid must be pooled 4x4 or the encoder grid')
        self.architecture_id='stream_rk_aligned';self.cache_contract='lip-rk-aligned-memory-v1'
        self.query_side=query_side;self.variant=f'R1K1_spatial{dense_side}_aligned{query_side}'
        self.aligned_readout=AlignedMemoryReadout(memory_frames,max_age,sigma)

    def forward(self,features,metadata,cache=None,profiler=None):
        if cache is not None and not isinstance(cache,AlignedCache):
            if not isinstance(cache,CrossCache) or cache.metadata:raise ValueError('Aligned tracker requires its own cache')
        result,next_cache,source,dense=self.forward_with_dense(features,metadata,cache,profiler)
        bank=getattr(cache,'anchors',None) or AnchorBank.empty(result['latent_object'],self.slots)
        patches=getattr(cache,'patches',None);geometry=getattr(cache,'patch_geometry',None)
        b=len(dense);n=self.dense_side**2
        if patches is None:patches=dense.new_zeros(b,self.slots,n,256)
        current_geometry=pool_memory_geometry(features['geometry'],self.dense_side)
        stored_geometry=current_geometry.new_zeros(b,self.slots,n,5) if geometry is None else geometry
        if stored_geometry.shape!=(b,self.slots,n,5):raise ValueError('Aligned geometry grid changed; reset stream')
        query=source[:,:16] if self.query_side==4 else dense
        query_geometry=pool_memory_geometry(features['geometry'],4) if self.query_side==4 else current_geometry
        residual,diagnostics=self.aligned_readout(result['latent_object'],query,query_geometry,patches,stored_geometry,bank,metadata,self.pose_condition(bank,features,metadata))
        result['latent']=result['latent']+residual
        delta=self.head(result['latent']).float()
        delta=delta*(1-.5*torch.tanh(self.update_strength)*(1-result['observation_support']))[:,None]
        with torch.autocast(dense.device.type,enabled=False):
            pose=update(features['T_base_centered'].float(),delta[:,:3],delta[:,3:],features['object_diameter_m'].float())
            result.update(pose_centered=pose,pose_original=original_pose(pose,features['mesh_center'].float()),delta_rotvec=delta[:,:3],delta_center_norm=delta[:,3:],
                aligned_update_norm=diagnostics['update_norm'],aligned_tokens_read=diagnostics['usable_tokens'],aligned_queries=diagnostics['usable_queries'])
        next_geometry,_=select_patches(bank,next_cache.anchors,geometry,None if geometry is None else geometry[...,4],current_geometry,current_geometry[...,4],metadata)
        return result,AlignedCache(next_cache.source,next_cache.contexts,anchors=next_cache.anchors,variant=self.variant,
            patches=next_cache.patches,patch_quality=next_cache.patch_quality,patch_geometry=next_geometry)
