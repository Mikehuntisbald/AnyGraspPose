"""Learned feedback towards a retained, supplied pose estimate.

The two coefficients are signed residual strengths, not probabilities. Zero
preserves the parent exactly; one replaces that action component by the action
to the reference. Rotation-vector interpolation is not SE(3) interpolation.
"""
import torch
from torch import nn
from lip.geometry.so3 import log
from lip.models.direct_pose_residual import apply_direct_residual


def reference_action(base, reference, diameter, *, detach_reference=True):
    """Reference in the existing camera-axis rotation / center-over-d units."""
    with torch.autocast(base.device.type, enabled=False):
        base, diameter = base.float(), diameter.float()
        reference = reference.detach().float() if detach_reference else reference.float()
        rotation = log(reference[:, :3, :3] @ base[:, :3, :3].transpose(-1, -2))
        center = (reference[:, :3, 3] - base[:, :3, 3]) / diameter[:, None]
        return torch.cat((rotation, center), -1)


class PoseReferenceFeedback(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_norm = nn.LayerNorm(256)
        self.state_norm = nn.LayerNorm(24)
        self.readout = nn.Sequential(nn.Linear(288, 128), nn.GELU(), nn.Linear(128, 2))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, latent, state, action_gap, support, age):
        b = len(latent)
        if latent.shape != (b, 256) or state.shape != (b, 24) or action_gap.shape != (b, 6):
            raise ValueError('Pose-reference feedback requires Bx256 latent, Bx24 state, Bx6 action gap')
        if support.shape != (b,) or age.shape != (b,):
            raise ValueError('Support and residence age must have shape B')
        condition = torch.cat((self.latent_norm(latent), self.state_norm(state),
            action_gap.float().clamp(-20, 20), support.float()[:, None],
            torch.log1p(age.float().clamp_min(0))[:, None]), -1)
        return torch.tanh(self.readout(condition))


def apply_reference_feedback(parent, features, target_action, coefficients):
    if coefficients.shape != (len(target_action), 2) or target_action.shape[-1] != 6:
        raise ValueError('Reference action / feedback coefficient shape mismatch')
    parent_action = torch.cat((parent['delta_rotvec'].float(), parent['delta_center_norm'].float()), -1)
    strengths = torch.repeat_interleave(coefficients.float(), 3, dim=-1)
    residual = strengths * (target_action.float() - parent_action)
    result = apply_direct_residual(parent, features, residual)
    result['reference_rotation_residual_norm'] = result.pop('direct_rotation_norm')
    result['reference_center_residual_norm'] = result.pop('direct_center_norm')
    result['reference_rotation_coefficient'] = coefficients[:, 0].float()
    result['reference_center_coefficient'] = coefficients[:, 1].float()
    return result
