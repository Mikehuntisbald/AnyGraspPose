"""Candidate-conditioned probability of frozen FP success, with calibrated logits."""
import torch
from torch import nn
from lip.geometry.so3 import exp,log,update

class BasinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,64),nn.GELU())
        self.candidate=nn.Sequential(nn.Linear(6,32),nn.GELU())
        self.score=nn.Sequential(nn.Linear(96,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        self.register_buffer('delta_scale',torch.tensor([torch.pi/12]*3+[.05]*3))
        self.register_buffer('temperature',torch.tensor(1.))
    def forward(self,z,delta):
        d=delta.float()/self.delta_scale
        return self.score(torch.cat((self.scene(z.float()),self.candidate(d)),dim=-1)).squeeze(-1)/self.temperature

def basin_loss(critic,z,delta):
    # Parameters are frozen by the caller, but derivatives w.r.t. action remain live.
    return torch.nn.functional.softplus(-critic(z.detach(),delta)).mean()

def candidate_poses(base,pred,diameter,generator):
    device=pred.device;dtype=pred.dtype
    def axis():
        x=torch.randn(3,generator=generator).to(device=device,dtype=dtype)
        return x/x.norm().clamp_min(1e-8)
    ra,ta=axis(),axis();rr,rt=axis(),axis()
    random_angle=float(torch.rand((),generator=generator))*torch.pi/6
    random_shift=float(torch.rand((),generator=generator))*.1
    perturb=[(0*ra,0*ta),(ra*torch.pi/36,ta*.02),(-ra*torch.pi/36,-ta*.02),
             (ra*torch.pi/12,0*ta),(0*ra,ta*.05),(rr*random_angle,rt*random_shift)]
    poses=[];deltas=[]
    for r,t in perturb:
        p=pred.clone();p[:3,:3]=exp(r)@pred[:3,:3];p[:3,3]+=diameter*t
        delta=torch.cat((log(p[:3,:3]@base[:3,:3].T),(p[:3,3]-base[:3,3])/diameter))
        poses.append(p);deltas.append(delta)
    return torch.stack(poses),torch.stack(deltas)

class BasinOutcomeCritic(nn.Module):
    """Dual probabilities and continuous post-FP error; candidate order uses quality."""
    def __init__(self):
        super().__init__()
        self.reference=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,6))
        self.reference.requires_grad_(False)
        self.scene=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,96),nn.GELU())
        self.action=nn.Sequential(nn.Linear(12,64),nn.GELU())
        self.trunk=nn.Sequential(nn.Linear(160,192),nn.GELU(),nn.Dropout(.1),nn.Linear(192,96),nn.GELU())
        self.quality=nn.Linear(96,1);nn.init.constant_(self.quality.bias,-4.)
        self.improve=nn.Linear(96,1)
        self.convergence_offset=nn.Linear(96,1);nn.init.zeros_(self.convergence_offset.weight);nn.init.zeros_(self.convergence_offset.bias)
        self.register_buffer('delta_scale',torch.tensor([torch.pi/12]*3+[.05]*3))
        self.register_buffer('temperatures',torch.ones(2))
        self.register_buffer('log_tau',torch.tensor(.1).log())
    def train(self,mode=True):
        super().train(mode);self.reference.eval();return self
    def outcomes(self,z,delta):
        z=z.float();delta=delta.float();anchor=self.reference(z)
        context=self.scene(z);action=self.action(torch.cat((delta/self.delta_scale,(delta-anchor)/self.delta_scale),-1))
        h=self.trunk(torch.cat((context,action),-1));quality=self.quality(h).squeeze(-1)
        converge=(self.log_tau-quality+self.convergence_offset(context).squeeze(-1))/self.temperatures[0]
        improve=self.improve(h).squeeze(-1)/self.temperatures[1]
        return dict(converge_logit=converge,improve_logit=improve,log_after=quality)
    def forward(self,z,delta):return self.outcomes(z,delta)['converge_logit']

def load_frozen_basin(path,device,expected_sha256=None,allow_experimental=False):
    import hashlib
    from pathlib import Path
    p=Path(path)
    if expected_sha256 and hashlib.sha256(p.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('Basin critic hash mismatch')
    ck=torch.load(p,map_location='cpu',weights_only=False)
    if not ck['receipt']['passed'] and not allow_experimental:raise ValueError('Basin critic failed quality gate; explicit experimental policy required')
    model=BasinOutcomeCritic() if ck.get('architecture')=='outcome_v2' else BasinCritic()
    model.load_state_dict(ck['model']);return model.to(device).eval().requires_grad_(False)
