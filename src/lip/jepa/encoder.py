"""Pinned local official DINOv2; no hub download during model construction."""
import hashlib
from pathlib import Path
import torch
from torch import nn


class FrozenDINO(nn.Module):
    def __init__(self, repo, checkpoint, expected_sha256=None, feature_layers=(6, 12)):
        super().__init__()
        self.set_feature_layers(feature_layers)
        checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if expected_sha256 and self.checkpoint_sha256 != expected_sha256:
            raise ValueError('DINO checkpoint hash mismatch')
        self.backbone = torch.hub.load(str(repo), 'dinov2_vits14_reg', source='local', pretrained=False)
        self.backbone.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.requires_grad_(False)
        self.eval()

    def set_feature_layers(self, layers):
        layers=tuple(layers)
        if len(layers)!=2 or any(type(x) is not int or not 1<=x<=12 for x in layers) or layers[0]>=layers[1]:
            raise ValueError("Expected two ascending 1-based DINO block indices")
        self.feature_layers=layers

    def train(self, mode=True):
        return super().train(False)

    def forward(self, rgb):
        with torch.set_grad_enabled(torch.is_grad_enabled() and getattr(self, "trainable_encoder", False)):
            mid, last = getattr(self, "graphed_features", getattr(self, "compiled_features", self.backbone.get_intermediate_layers))(rgb, n=[i-1 for i in self.feature_layers], reshape=False, norm=True)
        if mid.shape[1:] != (256, 384) or last.shape != mid.shape:
            raise RuntimeError('DINO CLS/register exclusion or patch contract changed')
        return mid, last
