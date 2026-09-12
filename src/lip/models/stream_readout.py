import torch
from torch import nn


class StreamReadout(nn.Module):
    def __init__(self,dual=False):
        super().__init__();self.dual=dual
        self.query=nn.Parameter(torch.empty(1,2 if dual else 1,256));nn.init.normal_(self.query,std=.02)
        if dual:
            self.context_projection=nn.Linear(256,256,bias=False)
            nn.init.zeros_(self.context_projection.weight)
            self.gate=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,1))
            nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-2.)

    def forward(self,z):
        if not self.dual:return dict(latent=z[:,0])
        obj,ctx=z.unbind(1);gate=torch.sigmoid(self.gate(torch.cat((obj,ctx),-1)))
        return dict(latent=obj+gate*self.context_projection(ctx),latent_object=obj,
                    latent_context=ctx,context_gate=gate) # gate is not confidence
