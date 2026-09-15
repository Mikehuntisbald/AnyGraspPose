"""Read a retained rotation anchor without erasing the adaptive reference."""
import torch
from torch import nn
from lip.geometry.so3 import exp, log


class RotationAnchorReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_norm=nn.LayerNorm(256);self.state_norm=nn.LayerNorm(24)
        self.readout=nn.Sequential(nn.Linear(288,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)

    def forward(self,latent,state,angular_evidence,support,age):
        batch=len(latent)
        if latent.shape!=(batch,256) or state.shape!=(batch,24) or angular_evidence.shape!=(batch,6):
            raise ValueError('Rotation anchor expects latent256, state24 and angular evidence6')
        if support.shape!=(batch,) or age.shape!=(batch,):raise ValueError('Rotation anchor support/age shape mismatch')
        condition=torch.cat((self.latent_norm(latent),self.state_norm(state),angular_evidence.float().clamp(-20,20),
            support.float()[:,None],torch.log1p(age.float().clamp_min(0))[:,None]),-1)
        # Fraction read from the initial anchor, not a calibrated probability.
        # Native clamp has a usable boundary gradient at the zero start.
        return self.readout(condition).float().clamp(0,1).squeeze(-1)


def angular_evidence(reference,anchor_rotation,visual_pose):
    with torch.autocast(reference.device.type,enabled=False):
        rotation=reference[:,:3,:3].float()
        visual=visual_pose.detach().float()
        valid=torch.isfinite(visual).flatten(1).all(1)&(visual[:,2,3]>0)
        observed=torch.where(valid[:,None,None],visual[:,:3,:3],rotation.detach())
        anchor=log(anchor_rotation.detach().float()@rotation.transpose(-1,-2))
        innovation=log(observed@rotation.transpose(-1,-2))
        return torch.cat((anchor,innovation),-1)


def blend_rotation_reference(reference,anchor_rotation,fraction):
    if anchor_rotation.shape!=(len(reference),3,3) or fraction.shape!=(len(reference),):
        raise ValueError('Rotation anchor blend shape mismatch')
    with torch.autocast(reference.device.type,enabled=False):
        reference=reference.float();rotation=reference[:,:3,:3]
        gap=log(anchor_rotation.detach().float()@rotation.transpose(-1,-2))
        increment=exp(fraction.float()[:,None]*gap)-torch.eye(3,device=reference.device,dtype=torch.float32)
        result=reference.clone()
        result[:,:3,:3]=rotation+increment@rotation
        return result
