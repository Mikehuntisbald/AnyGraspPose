"""Matched local flow readout: fixed feature evidence, explicit displacement.

No teacher input. Source weights initialize both arms identically; the control
zeros added evidence. The bounded correction remains +/-14 crop pixels so a
change cannot be credited to expanding its range.
"""
import torch
from torch import nn
from .cad_image_correspondence import sample_points

BASE_DIM=133
FEATURE_DIM=210

class LocalFlowHead(nn.Module):
    def __init__(self,use_extra=True):
        super().__init__()
        self.use_extra=use_extra
        self.net=nn.Sequential(nn.Linear(FEATURE_DIM,128),nn.GELU(),nn.Linear(128,2))

    @torch.no_grad()
    def initialize(self,endpoint):
        self.net[0].weight.zero_()
        self.net[0].weight[:,:BASE_DIM].copy_(endpoint[0].weight)
        self.net[0].bias.copy_(endpoint[0].bias)
        self.net[-1].weight.copy_(endpoint[-1].weight[:2])
        self.net[-1].bias.copy_(endpoint[-1].bias[:2])

    def forward(self,x):
        if not self.use_extra:x=torch.cat((x[...,:BASE_DIM],torch.zeros_like(x[...,BASE_DIM:])),dim=-1)
        return 14*self.net(x.float()).tanh()


def local_flow_features(base,coarse,reference,measured,scores,peak):
    # A 3x3 correlation neighborhood around the selected coarse-grid mode.
    offsets=torch.tensor([(x,y) for y in (-1,0,1) for x in (-1,0,1)],device=peak.device)
    xy=torch.stack((peak%16,peak//16),-1)[...,None,:]+offsets
    inside=((xy>=0)&(xy<16)).all(-1)
    ids=xy[...,1].clamp(0,15)*16+xy[...,0].clamp(0,15)
    local=scores.gather(-1,ids)
    local=(local-local.amax(-1,keepdim=True)).clamp(-10,0).masked_fill(~inside,-10)/10
    measured_map=measured.transpose(1,2).reshape(len(measured),64,16,16)
    observed=sample_points(measured_map,(coarse+.5)/14-.5)
    return torch.cat((base,coarse/224.,(coarse-reference['uv'])/56.,local,observed),-1)
