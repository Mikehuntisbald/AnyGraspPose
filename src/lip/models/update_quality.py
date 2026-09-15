"""Proposal-conditioned update quality; no FoundationPose or hand inputs."""
import torch
from torch import nn


class UpdateQualityHead(nn.Module):
    def __init__(self,observation_dim=815):
        super().__init__()
        self.register_buffer('observation_mean',torch.zeros(observation_dim))
        self.register_buffer('observation_std',torch.ones(observation_dim))
        self.register_buffer('action_std',torch.ones(6))
        self.observation=nn.Sequential(nn.Linear(observation_dim,256),nn.GELU(),nn.Linear(256,128),nn.GELU())
        self.action=nn.Sequential(nn.Linear(6,128),nn.GELU())
        self.joint=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,128),nn.GELU(),nn.Linear(128,2))

    def forward(self,observation,delta):
        x=((observation.float()-self.observation_mean)/self.observation_std).clamp(-10,10)
        d=(delta.float()/self.action_std).clamp(-10,10)
        x=self.observation(x);d=self.action(d);out=self.joint(torch.cat((x,d,x*d),-1))
        return dict(harm_logit=out[...,0],advantage=.05*out[...,1])


def observation_vector(features,output):
    from lip.models.stream_rk import observation_support
    tokens,frame=observation_support(features['geometry'])
    original_delta=torch.cat((output['delta_rotvec'],output['delta_center_norm']),-1)
    # Every feature is already available before the current GT is read.
    x=torch.cat((output['latent'],output['latent_object'],output['latent_context'],features['state_input'],tokens,frame[:,None],original_delta),-1)
    if x.shape[-1]!=815:raise ValueError('Update-quality observation contract changed')
    return x,original_delta
