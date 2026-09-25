"""Experimental pixel evidence from immutable sensor/reference inputs only.

No pose output, JEPA completion, or trainable student-encoder feature enters
this module. The static CAD descriptor is the frozen Utonia cache output.
"""
import torch
from torch import nn
from .conv_cross import ConvNormAct,ResidualConv
from .dpt_surface import FixedBilinear

class DenseMeasurementHead(nn.Module):
    def __init__(self,cad_dim):
        super().__init__()
        self.context=nn.Sequential(nn.LayerNorm(cad_dim),nn.Linear(cad_dim,32))
        self.stem=ConvNormAct(15,32,kernel=5)
        self.local=ResidualConv(32)
        self.down=nn.Sequential(ConvNormAct(32,64,stride=2),ResidualConv(64))
        self.up=FixedBilinear(112,224)
        self.output=nn.Sequential(ConvNormAct(96,32),nn.Conv2d(32,1,1))
        nn.init.constant_(self.output[-1].bias,-2.)
    def forward(self,rgb,reference,geometry,cad,available,bounds):
        reference=torch.where(available[:,None,None,None],reference,0.)
        geometry=torch.cat((geometry[:,:2],torch.where(available[:,None,None,None],geometry[:,2:],0.)),1)
        image=torch.where(bounds,torch.cat((rgb,reference,geometry),1),0.)
        safe_cad=torch.where(available[:,None],cad,0.)
        context=self.context(safe_cad)*available[:,None]
        local=self.local(self.stem(image)+context[:,:,None,None])
        value=self.output(torch.cat((local,self.up(self.down(local))),1)).float()
        return torch.where(bounds,value,-30.)
