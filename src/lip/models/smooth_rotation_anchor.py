"""Zero-start signed anchor read with no hard-clamp dead interval."""
import torch
from torch.nn import functional as F
from lip.models.rotation_anchor import RotationAnchorReadout
from lip.models.stream_rotation_anchor import RotationAnchorRKTracker


def signed_anchor_coefficient(logit):
    # FP32 avoids BF16 rounding in the denominator near the zero start.
    # Softsign is bounded (-1,1), with derivative 1/(1+|x|)^2.
    return F.softsign(logit.float())


class SmoothRotationAnchorReadout(RotationAnchorReadout):
    def forward(self,latent,state,angular_evidence,support,age):
        batch=len(latent)
        if latent.shape!=(batch,256) or state.shape!=(batch,24) or angular_evidence.shape!=(batch,6):
            raise ValueError('Rotation anchor expects latent256, state24 and angular evidence6')
        if support.shape!=(batch,) or age.shape!=(batch,):raise ValueError('Rotation anchor support/age shape mismatch')
        condition=torch.cat((self.latent_norm(latent),self.state_norm(state),angular_evidence.float().clamp(-20,20),
            support.float()[:,None],torch.log1p(age.float().clamp_min(0))[:,None]),-1)
        return signed_anchor_coefficient(self.readout(condition)).squeeze(-1)


class SmoothRotationAnchorRKTracker(RotationAnchorRKTracker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rotation_anchor_readout=SmoothRotationAnchorReadout()
        self.architecture_id='stream_rk_rotation_anchor_smooth'
        self.cache_contract='lip-rk-rotation-anchor-smooth-v1'
        self.variant=f'R1K1_spatial{self.dense_side}_rotation_anchor_smooth'

    def forward(self,*args,**kwargs):
        result,cache=super().forward(*args,**kwargs)
        # This coefficient may be negative: extrapolate away from a bad anchor.
        # It is neither a probability nor a convex-mixture fraction.
        result['rotation_anchor_coefficient']=result.pop('rotation_anchor_fraction')
        return result,cache
