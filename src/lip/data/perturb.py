import torch
from lip.geometry.so3 import exp


def noisy_history(poses, diameter, generator, enabled=True, rho=.9):
    if not enabled:return poses.clone(), dict(rotation_clipped=0., translation_clipped=0.)
    n=len(poses)
    k=int(torch.multinomial(torch.tensor([.75, .20, .05]), 1, generator=generator))
    std=torch.tensor([([2, 8, 20][k]*torch.pi/180)]*3+[([.01, .05, .10][k])]*3)
    bias=torch.randn(6, generator=generator)*std*(.5**.5)
    err=torch.randn(6, generator=generator)*std*(.5**.5)
    series=[]
    for _ in range(n):
        err=rho*err+(1-rho*rho)**.5*torch.randn(6, generator=generator)*std*(.5**.5)
        series.append(bias+err)
    e=torch.stack(series).to(poses.device);rn=e[:, :3].norm(dim=-1);tn=e[:, 3:].norm(dim=-1)
    e[:, :3]*=(torch.pi/4/rn.clamp_min(1e-8)).clamp_max(1)[:, None]
    e[:, 3:] *= (.25/tn.clamp_min(1e-8)).clamp_max(1)[:, None]
    out=poses.clone();out[:, :3, :3]=exp(e[:, :3]) @ poses[:, :3, :3]
    out[:, :3, 3]+=diameter*e[:, 3:]
    return out,dict(rotation_clipped=float((rn>torch.pi/4).float().mean()), translation_clipped=float((tn>.25).float().mean()))
