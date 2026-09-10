"""Optional adapter; imports no FoundationPose modules on the training path."""
import inspect
import contextlib
import torch


class FoundationPoseAdapter:
    def __init__(self, fp, device=None):
        self.fp=fp
        self.device=torch.device(device or (fp.pose_last.device if getattr(fp,'pose_last',None) is not None else 'cuda'))
        required={'rgb','depth','K','iteration'}
        if not required.issubset(inspect.signature(fp.track_one).parameters):
            raise TypeError('Unsupported FoundationPose track_one signature')

    def accept(self, pose_original):
        with torch.cuda.device(self.device) if self.device.type=='cuda' else contextlib.nullcontext():
            c=torch.as_tensor(self.fp.get_tf_to_centered_mesh(),dtype=torch.float32,device=self.device)
        pose=torch.as_tensor(pose_original,dtype=torch.float32,device=self.device)
        if pose.shape!=(4,4) or not torch.isfinite(pose).all():raise ValueError('Expected finite original-mesh 4x4 pose')
        self.fp.pose_last=(pose@torch.linalg.inv(c)).contiguous()

    def refine(self, prior_original, rgb, depth, K, iteration=2, accept_result=True):
        self.accept(prior_original)
        try:
            with torch.cuda.device(self.device) if self.device.type=='cuda' else contextlib.nullcontext():
                result=self.fp.track_one(rgb=rgb,depth=depth,K=K,iteration=iteration)
            accepted=result if accept_result else prior_original
            self.accept(accepted)
        except Exception:
            self.accept(prior_original);raise
        return accepted
