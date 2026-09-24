"""Frozen FP RGB-XYZ CNNs and a read-only identity for existing CAD caches."""
import hashlib
import importlib.util
import ast
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from lip.engine.jepa_checkpoint import sha

FP_SHA = '774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60'


class CachedUtonia(nn.Module):
    """Keeps CADStore's original content contract; cache misses never encode."""
    feature_dim = 1224
    fast = True

    def __init__(self, checkpoint, expected_sha):
        super().__init__()
        if sha(checkpoint) != expected_sha:
            raise ValueError('Static Utonia weight identity mismatch')
        self.checkpoint_sha256 = expected_sha
        package = Path(importlib.util.find_spec('utonia').origin).parent
        digest = hashlib.sha256()
        for path in sorted(package.rglob('*.py')):
            digest.update(str(path.relative_to(package)).encode()); digest.update(path.read_bytes())
        self.source_sha256 = digest.hexdigest()
        self.last_unrepresentable_clouds = []

    def forward(self, *args, **kwargs):
        raise RuntimeError('Missing sealed CAD cache: prepare with the pinned Utonia runtime; online Utonia is disabled')


class FrozenFPEncoder(nn.Module):
    def __init__(self, root, checkpoint, expected_sha=FP_SHA, size=160):
        super().__init__()
        if sha(checkpoint) != expected_sha:
            raise ValueError('FoundationPose weight identity mismatch')
        if size not in (160, 224):
            raise ValueError('Unsupported FP input size')
        from omegaconf import OmegaConf
        # Execute only official torch network definitions. Importing FP's GUI /
        # registration Utils also imports unrelated optional application packages.
        # The classes themselves are unmodified, and their source hashes are kept.
        scope=dict(torch=torch,nn=nn,F=F,math=math)
        selected={'ConvBNReLU','ResnetBasicBlock','conv3x3','PositionalEmbedding','RefineNet'}
        self.source_hashes={}
        for name in ('network_modules.py','refine_network.py'):
            path=Path(root)/'learning/models'/name
            self.source_hashes[name]=sha(path)
            nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in selected]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
        RefineNet=scope['RefineNet']
        config = OmegaConf.load(Path(checkpoint).with_name('config.yml'))
        if config.c_in != 6 or not config.normalize_xyz or config.use_normal or not config.use_BN:
            raise ValueError('Unexpected pretrained FP input convention')
        with torch.random.fork_rng(devices=[]):
            original = RefineNet(config, c_in=6)
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            original.load_state_dict(state.get('model', state), strict=True)
        # Names deliberately contain .encoder. so frozen weights remain external
        # to LIP's trainable checkpoint and are rebound by the weight receipt.
        self.encoder = nn.ModuleDict(dict(encodeA=original.encodeA, encodeAB=original.encodeAB))
        self.size = size
        self.checkpoint_sha256 = expected_sha
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @staticmethod
    def normalize_xyz(camera_xyz, depth, translation, diameter):
        xyz = (camera_xyz - translation[:, :, None, None]) / (diameter[:, None, None, None] * .5)
        invalid = (depth < .1) | ~torch.isfinite(depth)
        return torch.where(invalid | ~torch.isfinite(xyz) | (xyz.abs() >= 2), 0., xyz)

    @torch.no_grad()
    def encode_observation(self, rgb, camera_xyz, depth, translation, diameter):
        """Encode real RGB-XYZ only; no CAD image or encodeAB is consumed."""
        xyz = self.normalize_xyz(camera_xyz, depth, translation, diameter)
        if self.size != rgb.shape[-1]:
            rgb = F.interpolate(rgb, (self.size, self.size), mode='bilinear', align_corners=False)
            xyz = F.interpolate(xyz, (self.size, self.size), mode='nearest')
        value = torch.cat((rgb, xyz), 1).contiguous(memory_format=torch.channels_last)
        with torch.autocast(value.device.type, dtype=torch.bfloat16):
            observed = self.encoder['encodeA'](value)
        return F.adaptive_avg_pool2d(observed, (16, 16)).flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def forward(self, real_rgb, real_xyz, real_depth, cad_rgb, cad_xyz, cad_depth, translation, diameter):
        # Camera axes, object-radius scale, render first, observation second.
        xyz = self.normalize_xyz(torch.cat((cad_xyz, real_xyz)), torch.cat((cad_depth, real_depth)),
                                 translation.repeat(2, 1), diameter.repeat(2))
        rgb = torch.cat((cad_rgb, real_rgb))
        if self.size != rgb.shape[-1]:
            rgb = F.interpolate(rgb, (self.size, self.size), mode='bilinear', align_corners=False)
            xyz = F.interpolate(xyz, (self.size, self.size), mode='nearest')
        value = torch.cat((rgb, xyz), 1).contiguous(memory_format=torch.channels_last)
        with torch.autocast(value.device.type, dtype=torch.bfloat16):
            both = self.encoder['encodeA'](value)
            cad, observed = both.chunk(2)
            pair = self.encoder['encodeAB'](torch.cat((cad, observed), 1).contiguous(memory_format=torch.channels_last))
        pool = lambda x: F.adaptive_avg_pool2d(x, (16, 16)).flatten(2).transpose(1, 2).contiguous()
        return pool(observed), pool(pair)
