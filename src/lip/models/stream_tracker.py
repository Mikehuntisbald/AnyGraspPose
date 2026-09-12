"""Streaming single/dual readouts; legacy Tracker remains unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from lip.models.tracker import Tracker,spatial_position
from lip.models.stream_attention import StreamTemporal
from lip.models.stream_readout import StreamReadout
from lip.geometry.so3 import update,original_pose
from lip.engine.stream_state import CrossCache,TemporalCache,cache_contract_for


class StreamTracker(nn.Module):
    def __init__(self,architecture_id='stream_single',memory_frames=8,dropout=0.,pretrained=False,
                 time_unit=1/30,max_gap_seconds=.5,cache_kind='functional'):
        super().__init__()
        if architecture_id not in ('stream_single','stream_dual','stream_dual_cross'):raise ValueError('Unknown streaming architecture')
        if cache_kind not in ('functional','ring'):raise ValueError('Unknown cache implementation')
        if architecture_id=='stream_dual_cross' and cache_kind!='functional':raise ValueError('Cross context cache currently requires functional mode')
        self.architecture_id=architecture_id;self.memory_frames=memory_frames
        self.cache_contract=cache_contract_for(architecture_id)
        self.max_gap_seconds=max_gap_seconds;self.cache_kind=cache_kind;self.weights_version=0
        legacy=Tracker(pretrained,dropout)
        for name in ('rgb','rgb_proj','geometry','fusion','head'):setattr(self,name,getattr(legacy,name))
        self.token_type=nn.Parameter(legacy.token_type.detach().clone())
        self.state=nn.Sequential(nn.Linear(24,256),nn.GELU(),nn.Linear(256,256))
        self.source_position=nn.Linear(2,256)
        nn.init.zeros_(self.source_position.weight);nn.init.zeros_(self.source_position.bias)
        self.temporal=StreamTemporal(memory_frames,dropout,time_unit)
        if architecture_id=='stream_dual_cross':
            from lip.models.stream_cross_readout import StreamCrossReadout
            self.readout=StreamCrossReadout(memory_frames)
        else:self.readout=StreamReadout(architecture_id=='stream_dual')
        self.migration_status={};self.train()

    def train(self,mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        return self

    def load_state_dict(self,*args,**kwargs):
        result=super().load_state_dict(*args,**kwargs);self.weights_version+=1;return result

    def parameter_versions(self):
        return tuple(p._version for p in self.parameters())

    def encode_current(self,features,profiler=None):
        from contextlib import nullcontext
        scope=lambda name:profiler.section(name) if profiler is not None else nullcontext()
        rgb=features['rgb'];geometry=features['geometry']
        if rgb.ndim!=4 or geometry.ndim!=4:raise ValueError('Streaming encodes B current images, never a history image axis')
        with scope('rgb_encoder'):x=self.rgb_proj(self.rgb(rgb))
        with scope('geometry_encoder'):g=self.geometry(geometry)
        with scope('spatial_fusion'):
            side=x.shape[-1];pos=spatial_position(side,x.device).to(x.dtype)
            x=x.flatten(2).transpose(1,2)+pos;g=g.flatten(2).transpose(1,2)+pos
            for block in self.fusion:x=block(x,g)
            x=F.adaptive_avg_pool2d(x.transpose(1,2).reshape(len(rgb),256,side,side),4).flatten(2).transpose(1,2)
            x=x+spatial_position(4,x.device).to(x.dtype)+self.token_type[0]+self.source_position(features['source_xy'])
            state=self.state(features['state_input'])+self.token_type[1]
            return torch.cat((x,state[:,None]),1)

    def forward(self,features,metadata,cache=None,profiler=None):
        from contextlib import nullcontext
        scope=lambda name:profiler.section(name) if profiler is not None else nullcontext()
        source=self.encode_current(features,profiler)
        query=self.readout.query.expand(len(source),-1,-1)+self.token_type[2]
        with scope('temporal_readout'):
            if self.architecture_id=='stream_dual_cross':
                cache=cache or CrossCache(TemporalCache(capacity=self.memory_frames))
                if not isinstance(cache,CrossCache):raise ValueError('Cross architecture requires its own cache contract')
                z,next_source=self.temporal(source,query,metadata,cache.source,True)
                result,contexts=self.readout(z,metadata,cache.contexts)
                next_cache=CrossCache(next_source,contexts)
            else:
                if isinstance(cache,CrossCache):raise ValueError('Cannot use cross-context cache in a legacy architecture')
                z,next_cache=self.temporal(source,query,metadata,cache,self.readout.dual)
                result=self.readout(z)
            delta=self.head(result['latent']).float()
        with scope('pose_update'),torch.autocast(source.device.type,enabled=False):
            pose=update(features['T_base_centered'].float(),delta[:,:3],delta[:,3:],features['object_diameter_m'].float())
            result.update(pose_centered=pose,pose_original=original_pose(pose,features['mesh_center'].float()),
                delta_rotvec=delta[:,:3],delta_center_norm=delta[:,3:],confidence=None)
        return result,next_cache

    def initialize(self,T0_original,mesh,K,stream_id,timestamp0,**kwargs):
        from lip.engine.stream_runtime import initialize
        return initialize(self,T0_original,mesh,K,stream_id,timestamp0,**kwargs)

    @torch.no_grad()
    def step(self,rgb,depth,timestamp,state,**kwargs):
        from lip.engine.stream_runtime import step
        return step(self,rgb,depth,timestamp,state,**kwargs)

    def commit(self,proposal,next_state):
        from lip.engine.stream_runtime import commit
        return commit(self,proposal,next_state)

    def correct(self,state,pose_original,timestamp=None,relocalization=False):
        from lip.engine.stream_runtime import correct
        return correct(self,state,pose_original,timestamp,relocalization)
