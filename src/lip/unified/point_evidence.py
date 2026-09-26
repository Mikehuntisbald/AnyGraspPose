"""Point evidence reads observed and reconstructed information separately.

The V58 stage freezes every V56 parameter. This head cannot improve its labels
by moving flow endpoints or changing reconstructed depth. Labels are defined in
a separate function and never enter inference.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .cad_image_correspondence import sample_points

BASE_DIM=139
FEATURE_DIM=272


class PointEvidenceHead(nn.Module):
    def __init__(self,use_observation=True):
        super().__init__()
        self.use_observation=use_observation
        self.net=nn.Sequential(nn.Linear(FEATURE_DIM,128),nn.GELU(),nn.LayerNorm(128),
                               nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))

    def forward(self,features):
        if not self.use_observation:
            features=torch.cat((features[...,:BASE_DIM],torch.zeros_like(features[...,BASE_DIM:])),dim=-1)
        return self.net(features.float())


@torch.no_grad()
def point_evidence_features(source,current,observed,reference,uv,geometry,recovered,entropy,logits):
    b=len(source)
    def descriptor_at(value):
        grid=value.transpose(1,2).reshape(b,64,16,16)
        return sample_points(grid,(uv+.5)/14-.5)
    restored=descriptor_at(current);real=descriptor_at(observed)
    base=torch.cat((source,restored,reference['xyz'],reference['uv']/224.,uv/224.,
                    reference['depth'],entropy[...,None],logits.sigmoid()),dim=-1)
    measured=sample_points(geometry[:,:2],uv,mode='nearest')
    depth=reference['depth'] if recovered is None else sample_points(recovered[:,3:4],uv)
    has_recovery=torch.full_like(depth,float(recovered is not None))
    extra=torch.cat((real,real-restored,measured,depth,(measured[:,:,:1]-depth)*measured[:,:,1:2],has_recovery),dim=-1)
    result=torch.cat((base,extra),dim=-1)
    if base.shape[-1]!=BASE_DIM or result.shape[-1]!=FEATURE_DIM:raise RuntimeError('Evidence feature contract changed')
    return result.detach()


@torch.no_grad()
def point_evidence_targets(output,target,observation,visible,crop_k):
    from .flow_reconstruction import flow_labels
    labels=flow_labels(output,target,crop_k,observation.diameter,visible)
    uv=output['flow_rounds'][-1]['uv'].detach()
    eroded=-F.max_pool2d(-visible.float(),3,1,1)>.999
    real_pixel=sample_points(eroded.float(),uv,'nearest')[...,0]>.5
    measured=sample_points(observation.measured_depth_m,uv,'nearest')[...,0]
    cad_depth=sample_points(target.cad_geometry_depth_m,uv,'nearest')[...,0]
    inside=(uv>=2).all(-1)&(uv<222).all(-1)
    # This is a new, explicitly named measurement-usability target. Existing
    # real depth reconstruction labels are neither replaced nor filtered by it.
    quality=(labels['visible']&real_pixel&inside&torch.isfinite(measured)&(measured>0)
             &((uv-labels['uv']).norm(dim=-1)<=3.)&((measured-cad_depth).abs()<=.03))
    known=torch.stack((labels['known_visible'],labels['known_support']),-1)
    values=torch.stack((labels['visible'],quality),-1).float().masked_fill(~known,float('nan'))
    return values,labels
