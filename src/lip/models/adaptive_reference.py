"""Bounded causal writing of a pose reference from a detached visual proposal.

The writer is trained by later pose losses through its reference state. It
does not differentiate through past accepted poses or the proposal being
stored. Gates start exactly at zero; clamp uses its native boundary gradient.
"""
import torch
from torch import nn
from lip.geometry.so3 import exp
from lip.models.pose_reference import reference_action


class ReferenceWriter(nn.Module):
    def __init__(self, limit=.25):
        super().__init__()
        if not 0<limit<=1:raise ValueError('Reference write limit must be in (0,1]')
        self.limit=float(limit)
        self.latent_norm=nn.LayerNorm(256);self.state_norm=nn.LayerNorm(24)
        self.readout=nn.Sequential(nn.Linear(288,128),nn.GELU(),nn.Linear(128,2))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)

    def forward(self,latent,state,innovation,support,age):
        b=len(latent)
        if latent.shape!=(b,256) or state.shape!=(b,24) or innovation.shape!=(b,6) or support.shape!=(b,) or age.shape!=(b,):
            raise ValueError('Reference writer expects latent256, state24, innovation6, support and age')
        condition=torch.cat((self.latent_norm(latent),self.state_norm(state),innovation.float().clamp(-20,20),
            support.float()[:,None],torch.log1p(age.float().clamp_min(0))[:,None]),-1)
        # Values are in [0, limit]. They are write fractions, not calibrated
        # probabilities. At exactly zero native clamp retains a subgradient;
        # strictly negative or saturated logits have zero gradient.
        return self.limit*self.readout(condition).float().clamp(0,1)


def write_innovation(reference,measurement,diameter):
    with torch.autocast(reference.device.type,enabled=False):
        measurement=measurement.detach().float()
        valid=torch.isfinite(measurement).flatten(1).all(1)&(measurement[:,2,3]>0)
        safe=torch.where(valid[:,None,None],measurement,reference.detach().float())
        innovation=reference_action(reference,safe,diameter)
        return innovation,valid


def write_reference(reference,innovation,gates,diameter):
    """Left SO(3) interpolation and a convex camera-center update, in FP32."""
    if innovation.shape!=(len(reference),6) or gates.shape!=(len(reference),2):
        raise ValueError('Reference write shape mismatch')
    with torch.autocast(reference.device.type,enabled=False):
        reference=reference.float();innovation=innovation.float();gates=gates.float();diameter=diameter.float()
        rotation=innovation[:,:3]*gates[:,0:1];center=innovation[:,3:]*gates[:,1:2]
        delta_rotation=exp(rotation)-torch.eye(3,device=reference.device,dtype=torch.float32)
        result=reference.clone()
        # This equivalent form preserves the exact parent at zero, including
        # on matmul backends that round inputs to an identity multiplication.
        result[:,:3,:3]=reference[:,:3,:3]+delta_rotation@reference[:,:3,:3]
        result[:,:3,3]=reference[:,:3,3]+diameter[:,None]*center
        return result,rotation.norm(dim=-1),center.norm(dim=-1)
