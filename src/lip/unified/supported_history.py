"""Preserve learned memory features; accept sparse confident object support.

Foreground confidence and crop-cell occupancy are different quantities. A cell
containing one confident observed patch is useful even when 15 patches are BG.
"""
from .model import SpatialWriter
from .conv_cross import ConvCrossTracker


class SupportedSpatialWriter(SpatialWriter):
    def object_weight(self,visible,valid):
        # Use confident measured image patches; absent depth does not erase RGB.
        # The inherited writer separately tracks validity of the XYZ coordinate.
        return visible*valid*(visible>=.5)

    def object_reliable(self,mass):
        # >= one selected patch contributes at least .5 / 16. No fabricated point.
        return mass>0


class SupportedHistoryTracker(ConvCrossTracker):
    architecture_id='stream_conv_cross_supported_history_jepa_v10'
    model_version='conv-cross-state-supported-history-v10'
    cache_contract='observed-confident-support128-v10'

    def __init__(self,encoder,cached_utonia):
        super().__init__(encoder,cached_utonia)
        writer=SupportedSpatialWriter()
        writer.load_state_dict(self.writer.state_dict(),strict=True)
        self.writer=writer
        self.weights_version=self.model_version+'/initial'
        self.migration.update(history='global QK; confident observed support; original learned writer mapping')
