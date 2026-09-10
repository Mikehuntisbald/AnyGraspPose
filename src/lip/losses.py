import torch
from lip.geometry.so3 import angle


def pose_loss(pred, target, points, diameter, valid=None):
    with torch.autocast(pred.device.type, enabled=False):
        pred, target, points, diameter=[x.float() for x in (pred, target, points, diameter)]
        trans=torch.nn.functional.smooth_l1_loss((pred[:, :3, 3]-target[:, :3, 3])/diameter[:, None],
                                                  torch.zeros_like(pred[:, :3, 3]), beta=.02, reduction='none').mean(-1)
        rot=angle(pred[:, :3, :3] @ target[:, :3, :3].transpose(-1, -2))
        p=points @ pred[:, :3, :3].transpose(-1, -2)+pred[:, None, :3, 3]
        t=points @ target[:, :3, :3].transpose(-1, -2)+target[:, None, :3, 3]
        pts=torch.linalg.vector_norm(p-t, dim=-1).mean(-1)/diameter
        valid=torch.ones_like(trans) if valid is None else valid.float()
        mean=lambda v:(v*valid).sum()/valid.sum().clamp_min(1)
        return mean(trans+.5*rot+pts),dict(translation=mean(trans), rotation=mean(rot), points=mean(pts))
