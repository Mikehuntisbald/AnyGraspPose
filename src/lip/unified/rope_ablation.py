"""Frozen, reversible RoPE-gate intervention for paired inference only."""
from contextlib import contextmanager
import hashlib
import torch


def state_digest(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenRoPESwitch:
    def __init__(self,model):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('RoPE ablation requires frozen evaluation weights')
        if not getattr(model,'disable_history',False):
            raise ValueError('This paired probe requires history disabled in both arms')
        self.gain=model.cad_surface.rope3d.gain
        self.original=self.gain.detach().clone()
        self.before=state_digest(model)

    @contextmanager
    def arm(self,enabled):
        if torch.is_grad_enabled():raise ValueError('Inference-only intervention requires no_grad')
        try:
            self.gain.copy_(self.original if enabled else torch.zeros_like(self.original))
            yield
        finally:
            self.gain.copy_(self.original)

    def verify(self,model):
        if state_digest(model)!=self.before:
            raise ValueError('Frozen ablation changed model state')
        return dict(model_state_unchanged=True,state_sha256=self.before,
                    original_gains=self.original.cpu().tolist(),effective_gains=self.original.tanh().cpu().tolist())
