"""Dense current-observation to CAD geometry matching, with a zero-start rotation residual."""
import torch
from torch import nn
from lip.models.stream_tracker import StreamTracker
from lip.models.tracker import CrossBlock,spatial_position
from lip.geometry.so3 import exp,log,original_pose


class RotationAlignment(nn.Module):
    def __init__(self,use_parent_latent=True):
        super().__init__()
        self.use_parent_latent=use_parent_latent
        # Only CAD-rendered channels: depth, silhouette, and object XYZ. No GT.
        self.cad=nn.Sequential(nn.Conv2d(5,64,4,4),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv2d(64,256,4,4),nn.GroupNorm(8,256),nn.GELU())
        self.match=CrossBlock(0.)
        self.query=nn.Parameter(torch.randn(1,1,256)*.02)
        self.read=nn.MultiheadAttention(256,8,batch_first=True)
        self.output=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU(),nn.Linear(256,3))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,dense,geometry,latent,return_matching=False):
        if dense.shape[1:]!=(196,256) or geometry.shape[-2:]!=(224,224):raise ValueError('Rotation alignment requires 14x14 / 224 inputs')
        cad=self.cad(geometry[:,2:7]).flatten(2).transpose(1,2)
        cad=cad+spatial_position(14,cad.device).to(cad.dtype)
        matched=self.match(dense,cad)
        read,_=self.read(self.query.expand(len(dense),-1,-1),matched,matched,need_weights=False)
        correction=self.output(torch.cat((read[:,0],latent if self.use_parent_latent else torch.zeros_like(latent)),-1))
        if return_matching:
            # Reuse the actual match Q/K projections for supervision. The normal
            # SDPA forward remains identical; no alternative attention output.
            return correction,(self.match.qnorm(dense),self.match.knorm(cad))
        return correction

    def correspondence_log_probability(self,query,key):
        """Log of mean-head attention probability, computed in FP32 for loss only."""
        with torch.autocast(query.device.type,enabled=False):
            w=self.match.attn.in_proj_weight.float();bias=self.match.attn.in_proj_bias.float()
            q=torch.nn.functional.linear(query.float(),w[:256],bias[:256])
            k=torch.nn.functional.linear(key.float(),w[256:512],bias[256:512])
            q=q.reshape(len(q),196,8,32).transpose(1,2);k=k.reshape(len(k),196,8,32).transpose(1,2)
            per_head=(q@k.transpose(-1,-2)/(32**.5)).log_softmax(-1)
            return torch.logsumexp(per_head,dim=1)-__import__('math').log(8)


class RotationAlignmentTracker(StreamTracker):
    def __init__(self,use_parent_latent=True,alignment_supervision=False,**kwargs):
        super().__init__(architecture_id='stream_dual_cross_residual',**kwargs)
        self.rotation_alignment=RotationAlignment(use_parent_latent)
        self.alignment_supervision=alignment_supervision

    def forward(self,features,metadata,cache=None,profiler=None):
        source,dense=self.encode_current(features,profiler,return_dense=True)
        result,next_cache=self.forward_encoded(source,features,metadata,cache,profiler)
        if self.alignment_supervision and self.training:
            correction,matching=self.rotation_alignment(dense,features['geometry'],result['latent'],return_matching=True)
            result['alignment_matching_inputs']=matching
        else:correction=self.rotation_alignment(dense,features['geometry'],result['latent'])
        correction=correction.float()
        with torch.autocast(correction.device.type,enabled=False):
            pose=result['pose_centered'].clone()
            pose[:,:3,:3]=exp(correction)@pose[:,:3,:3].clone()
            result.update(pose_centered=pose,pose_original=original_pose(pose,features['mesh_center'].float()),
                delta_rotvec=log(exp(correction)@exp(result['delta_rotvec'].float())),rotation_alignment_rotvec=correction)
        return result,next_cache
