"""Keep 16 spatially diverse real foreground patches instead of averaging cells."""
import torch
from torch import nn
from .model import SpatialWriter,Record,local_pool
from .conv_cross import ConvCrossTracker
from lip.jepa.positions import source_coordinates


def diverse_patches(confidence,valid,xy,count=16):
    """Fixed-budget confidence-weighted farthest sampling; no CPU/GPU sync."""
    with torch.no_grad():
        available=valid&(confidence>=.5)
        distance=torch.full_like(confidence,1e6)
        indices=[];masks=[]
        for i in range(count):
            score=confidence if i==0 else distance*confidence
            index=score.masked_fill(~available,-1).argmax(-1,keepdim=True)
            supported=available.gather(1,index)
            indices.append(index);masks.append(supported)
            point=xy.gather(1,index[...,None].expand(-1,-1,2))
            distance=torch.minimum(distance,(xy-point).square().sum(-1))
            available=available.scatter(1,index,False)
    return torch.cat(indices,1),torch.cat(masks,1)


class DenseObservedWriter(SpatialWriter):
    def __init__(self):
        super().__init__()
        grid=torch.stack((torch.arange(256)%16,torch.arange(256)//16),-1).float()
        self.register_buffer('selection_grid',grid,persistent=False)

    def forward(self,h,position,visible,valid,observation,memory):
        source=self.proj(h+position);source=source+self.ff(source)
        xy=source_coordinates(observation.packet.A_image_to_crop,observation.packet.source_size_wh)
        indices,reliable=diverse_patches(visible,valid,self.selection_grid[None].expand(len(h),-1,-1))
        def take(value):return value.gather(1,indices[...,None].expand(-1,-1,value.shape[-1]))
        depth_valid=observation.depth_valid.gather(1,indices)&reliable
        quality=(visible*valid).sum(-1)/valid.sum(-1).clamp_min(1)
        obj=Record(take(source),reliable,take(xy),take(observation.object_xyz),depth_valid,observation.packet.timestamp_s,quality)
        features,mass=local_pool(source,valid.float());coordinates,_=local_pool(xy,valid.float())
        xyz,depth_mass=local_pool(observation.object_xyz,valid.float()*observation.depth_valid)
        ctx=Record(features,mass>0,coordinates,xyz,depth_mass>0,observation.packet.timestamp_s,quality)
        return self.commit(obj,ctx,observation,memory)


class DenseHistoryTracker(ConvCrossTracker):
    architecture_id='stream_conv_cross_dense_history_jepa_v10'
    model_version='conv-cross-state-dense-observed-history-v10'
    cache_contract='observed-diverse16-global-qk128-v10'

    def __init__(self,encoder,cached_utonia):
        super().__init__(encoder,cached_utonia)
        writer=DenseObservedWriter();writer.load_state_dict(self.writer.state_dict(),strict=True);self.writer=writer
        self.weights_version=self.model_version+'/initial'
        self.migration.update(history='16 diverse real foreground patches per object record; global QK; preserved writer mapping')
