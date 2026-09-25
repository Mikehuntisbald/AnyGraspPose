"""Experimental JEPA-to-CAD coordinate readout; not enabled in production.

The dense queries come exclusively from DPT's JEPA inputs. CAD descriptors and
dynamic geometry are keys; actual canonical CAD points are values. This pilot
tests correspondence recovery before integration into the pose tracker.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .dpt_surface import DPTSurfaceHead, FixedBilinear


class CADCoordinateDecoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.dpt = DPTSurfaceHead()
        self.query = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 1))
        self.descriptor = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 32))
        self.geometry = nn.Sequential(nn.Linear(21, 64), nn.GELU(), nn.Linear(64, 32))
        self.offset = nn.Conv2d(16, 3, 3, padding=1)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        self.resize = FixedBilinear(56, 224)
        y,x=torch.meshgrid(torch.arange(56),torch.arange(56),indexing='ij')
        self.register_buffer('uv',torch.stack(((x.flatten()+.5)/56,(y.flatten()+.5)/56),-1))

    def forward(self, levels, valid, features, geometry, available):
        dense=self.dpt.dense_features(levels,valid)
        old=self.dpt.output[-1](dense).float()
        safe_features=torch.where(available[...,None],features,0.)
        safe_geometry=torch.where(available[...,None],geometry,0.)
        query=self.query(F.avg_pool2d(dense,4,4)).flatten(2).transpose(1,2)
        key=self.descriptor(safe_features)+self.geometry(safe_geometry)
        logits=(query@key.transpose(1,2)).float()/math.sqrt(32)
        distance=(self.uv[None,:,None]-safe_geometry[:,None,:,15:17]).square().sum(-1)
        logits=logits-2*distance.clamp_max(8)
        logits=logits.masked_fill(~available[:,None],-1e4)
        probability=logits.softmax(-1)*available[:,None]
        probability=probability/probability.sum(-1,keepdim=True).clamp_min(1e-8)
        anchor=(probability@safe_geometry[:,:,:3].float()).transpose(1,2).reshape(-1,3,56,56)
        xyz=self.resize(anchor)+.1*self.offset(dense).float().tanh()
        xyz=torch.where(available.any(-1)[:,None,None,None],xyz,old[:,:3])
        return torch.cat((xyz,old[:,3:]),1),logits


def dense_anchor_loss(logits,xyz,mask,cad_xyz,available):
    """Nearest-anchor CE in the original texture-aware canonical gauge.

    Do not independently choose a different symmetry representative per pixel.
    Low-purity cells are excluded rather than averaging two surfaces as a target.
    """
    mass=F.avg_pool2d(mask.float(),4,4)
    target=F.avg_pool2d(xyz*mask,4,4)/mass.clamp_min(1e-6)
    target=target.flatten(2).transpose(1,2)
    with torch.no_grad():
        distance=torch.cdist(target.float(),cad_xyz.float()).masked_fill(~available[:,None],float('inf'))
        labels=distance.argmin(-1)
        eligible=(mass.flatten(1)>=.9)&available.any(-1)[:,None]
    loss=F.cross_entropy(logits.transpose(1,2),labels,reduction='none')
    return (loss*eligible).sum()/eligible.sum().clamp_min(1)
