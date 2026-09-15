"""Keep trained spatial memory and add a direct action residual readout."""
from lip.models.stream_spatial_memory import SpatialRKTracker
from lip.models.direct_pose_residual import DirectPoseResidual,apply_direct_residual


class DirectPoseRKTracker(SpatialRKTracker):
    def __init__(self,memory_frames=8,dropout=0.,time_unit=1/30,max_gap_seconds=.5,
                 slots=4,min_gap=4,max_age=64,tolerance=.05,dense_side=14):
        super().__init__(memory_frames,dropout,time_unit,max_gap_seconds,slots,min_gap,max_age,tolerance,dense_side)
        self.architecture_id='stream_rk_direct_pose'
        self.cache_contract='lip-rk-spatial-direct-pose-v1'
        self.variant=f'R1K1_spatial{dense_side}_direct_pose'
        self.direct_pose_residual=DirectPoseResidual()

    def forward(self,features,metadata,cache=None,profiler=None):
        parent,next_cache=super().forward(features,metadata,cache,profiler)
        residual=self.direct_pose_residual(parent['latent'])
        return apply_direct_residual(parent,features,residual),next_cache
