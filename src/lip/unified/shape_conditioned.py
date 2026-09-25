"""Shape-normalized relation feature for the learned object-query readout.

This is NOT a pose output or a correction applied to the pose. It conditions a
learned relation token using the same restored/observed correspondences already
read by CompletionRelations; no raw JEPA, FP or teacher shortcut is introduced.
"""
import torch


def normalized_relation(scatter, covariance, mean_p, mean_y):
    with torch.autocast(scatter.device.type, enabled=False):
        scatter=scatter.float();covariance=covariance.float()
        scale=scatter.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-6)
        identity=torch.eye(3,device=scatter.device)[None]
        information=identity-scatter/scale[:,None,None]
        torque=torch.stack((covariance[:,1,2]-covariance[:,2,1],
                            covariance[:,2,0]-covariance[:,0,2],
                            covariance[:,0,1]-covariance[:,1,0]),-1)/scale[:,None]
        # Damping prevents an unobservable axis from acquiring infinite gain.
        omega=torch.linalg.solve(information+1e-3*identity,torque[...,None]).squeeze(-1)
        translation=mean_y.float()-mean_p.float()-torch.linalg.cross(omega,mean_p.float())
        signal=torch.cat((omega/.1745329252,translation/.05),-1)
        return signal.clamp(-3,3)
