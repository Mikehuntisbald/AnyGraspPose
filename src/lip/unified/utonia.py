"""Pinned, frozen point encoder with deterministic, shared-coordinate sampling."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from lip.engine.jepa_checkpoint import sha


class FrozenUtonia(nn.Module):
    def __init__(self, checkpoint, expected_sha, fast=False):
        super().__init__()
        if sha(checkpoint) != expected_sha:
            raise ValueError('Utonia weight identity mismatch')
        import utonia
        package=Path(utonia.__file__).parent;source=hashlib.sha256()
        for path in sorted(package.rglob('*.py')):
            source.update(str(path.relative_to(package)).encode());source.update(path.read_bytes())
        self.source_sha256=source.hexdigest()
        self.encoder = utonia.model.load(str(checkpoint), custom_config={'shuffle_orders': False})
        # Pooling modules have their own shuffle_orders default, independent
        # of the top-level config. Disable every serialization permutation.
        for module in self.encoder.modules():
            if hasattr(module, 'shuffle_orders'):module.shuffle_orders=False
        self.fast=fast
        if fast:
            from .utonia_fast import install
            install(self.encoder)
        config = torch.load(checkpoint, map_location='cpu', weights_only=True)['config']
        self.feature_dim = sum(config['enc_channels'][-3:])
        self.checkpoint_sha256 = expected_sha
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, clouds):
        """cloud: coord in common object units, color[0,1], normal; never GT."""
        result = [None] * len(clouds)
        self.last_unrepresentable_clouds=[]
        selected, coords, grids, features, batches, inverses = [], [], [], [], [], []
        offset = 0
        for lane, cloud in enumerate(clouds):
            xyz = cloud['coord'].float()
            if not len(xyz):
                continue
            if not all(torch.isfinite(cloud[k]).all() for k in ('coord', 'color', 'normal')):
                raise ValueError('Nonfinite point observation')
            grid = torch.floor(xyz / .01).to(torch.int64)
            minimum=grid.amin(0);span=grid.amax(0)-minimum
            if int(span.max())>=2**16:
                # Utonia's serialization supports at most 16 bits per axis.
                # Do not rescale coordinates or invent depth to satisfy it.
                # Returning None masks this lane's observed geometry tokens;
                # RGB, CAD geometry and history remain available.
                self.last_unrepresentable_clouds.append(dict(lane=lane,grid_span=span.tolist()))
                continue
            if self.fast:
                from .utonia_fast import unique_grid
                unique,inverse,_=unique_grid(grid-minimum);unique=unique+minimum
            else:
                unique,inverse=torch.unique(grid,dim=0,sorted=True,return_inverse=True)
            first = torch.full((len(unique),), len(xyz), device=xyz.device, dtype=torch.long)
            first.scatter_reduce_(0, inverse, torch.arange(len(xyz), device=xyz.device), reduce='amin')
            coords.append(xyz[first]); grids.append((unique - unique.min(0).values).int())
            features.append(torch.cat((xyz[first], cloud['color'][first].float()*2-1, cloud['normal'][first].float()), -1))
            batches.append(torch.full((len(first),), len(selected), device=xyz.device, dtype=torch.long))
            inverses.append(inverse + offset); offset += len(first); selected.append(lane)
        if not selected:
            return result
        with torch.autocast(coords[0].device.type, enabled=False):
            point = self.encoder(dict(coord=torch.cat(coords), grid_coord=torch.cat(grids),
                feat=torch.cat(features), batch=torch.cat(batches), grid_size=.01))
            for _ in range(2):
                parent = point.pop('pooling_parent'); inverse = point.pop('pooling_inverse')
                parent.feat = torch.cat((parent.feat, point.feat[inverse]), -1); point = parent
            while 'pooling_parent' in point:
                parent = point.pop('pooling_parent'); inverse = point.pop('pooling_inverse')
                parent.feat = point.feat[inverse]; point = parent
            for lane, inverse in zip(selected, inverses):
                result[lane] = point.feat[inverse].float()
        if any(x is not None and (x.shape[-1] != self.feature_dim or not torch.isfinite(x).all()) for x in result):
            raise ValueError('Utonia feature contract failed')
        return result
