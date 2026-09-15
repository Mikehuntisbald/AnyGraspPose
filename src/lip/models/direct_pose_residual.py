"""Separate rotation/center residuals in the existing camera-axis action space.

Residuals are added to the parent's action before the same
Update(T_base, delta); addition of rotation vectors is not SE(3) composition.
"""
import torch
from torch import nn
from lip.geometry.so3 import update, original_pose


class DirectPoseResidual(nn.Module):
    def __init__(self):
        super().__init__()
        def branch():
            layers=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128),nn.GELU(),nn.Linear(128,3))
            nn.init.zeros_(layers[-1].weight);nn.init.zeros_(layers[-1].bias)
            return layers
        self.rotation=branch()
        self.center=branch()

    def forward(self,latent):
        if latent.ndim!=2 or latent.shape[-1]!=256:
            raise ValueError('Direct pose residual expects B x 256 latent')
        return torch.cat((self.rotation(latent),self.center(latent)),-1)


def apply_direct_residual(parent,features,residual):
    """Reuse the parent's base and canonical decoupled pose update in FP32."""
    if residual.shape!=(*parent['delta_rotvec'].shape[:-1],6):
        raise ValueError('Residual action shape differs from the parent proposal')
    with torch.autocast(residual.device.type,enabled=False):
        delta=torch.cat((parent['delta_rotvec'].float(),parent['delta_center_norm'].float()),-1)+residual.float()
        pose=update(features['T_base_centered'].float(),delta[:,:3],delta[:,3:],features['object_diameter_m'].float())
        result=dict(parent,pose_centered=pose,pose_original=original_pose(pose,features['mesh_center'].float()),
            delta_rotvec=delta[:,:3],delta_center_norm=delta[:,3:],direct_rotation_norm=residual[:,:3].float().norm(dim=-1),
            direct_center_norm=residual[:,3:].float().norm(dim=-1))
    return result
