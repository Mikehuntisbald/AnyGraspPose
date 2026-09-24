"""Optional rigid-only CAD appearance bank; no sequence images in this cache."""
import hashlib
import json
from pathlib import Path
import time
import shlex
import inspect
import fcntl
import os
import numpy as np
import torch
from torch.nn import functional as F
from lip.data.jepa_pairs import normalize_rgb
from lip.jepa.positions import sample_features
from lip.jepa.rigid_solver import project_and_jacobian


def cad_asset_hash(mesh_path):
    """Bind geometry, referenced material/texture files and the actual renderer source."""
    mesh_path = Path(mesh_path)
    paths = {mesh_path}
    for line in mesh_path.read_text().splitlines():
        if line.startswith('mtllib '):
            for name in shlex.split(line)[1:]:
                material = mesh_path.parent / name
                paths.add(material)
                for row in material.read_text().splitlines():
                    if row.strip().startswith('map_Kd '):
                        paths.add(material.parent / shlex.split(row)[-1])
    from lip.geometry.appearance_renderer import AppearanceRenderer
    from lip.geometry.renderer import Renderer
    paths.update(Path(inspect.getsourcefile(cls)) for cls in (AppearanceRenderer, Renderer))
    content = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def sample_pixels(image, xy, mode='nearest'):
    h, w = image.shape[-2:]
    grid = torch.stack((2 * (xy[:, 0] + .5) / w - 1, 2 * (xy[:, 1] + .5) / h - 1), -1)
    return F.grid_sample(image[None].float(), grid[None, :, None], mode=mode, align_corners=False)[0, :, :, 0].T


class CADAnchorBank:
    def __init__(self, points, features, valid, view_directions, manifest):
        self.points = points
        self.features = features
        self.valid = valid
        self.view_directions = view_directions
        self.manifest = manifest

    def reference(self, base):
        direction = -(base[:3, :3].T @ base[:3, 3])
        direction = F.normalize(direction, dim=0)
        score = (self.view_directions @ direction)[:, None].expand_as(self.valid).masked_fill(~self.valid, -2.)
        view = score.argmax(0)
        index = torch.arange(len(self.points), device=base.device)
        return self.features[view, index], self.valid.any(0)

    @classmethod
    @torch.no_grad()
    def create(cls, mesh, appearance, renderer, encoder, cache_dir, mesh_hash):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_key = hashlib.sha256((mesh_hash + encoder.checkpoint_sha256).encode()).hexdigest()
        with (cache_dir / (lock_key + '.lock')).open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return cls._create_locked(mesh, appearance, renderer, encoder, cache_dir, mesh_hash)

    @classmethod
    @torch.no_grad()
    def _create_locked(cls, mesh, appearance, renderer, encoder, cache_dir, mesh_hash):
        from scipy.spatial.transform import Rotation
        key = hashlib.sha256((mesh_hash + encoder.checkpoint_sha256 + 'anchors256/views12/seed42/RGB224/ImageNet/BF16/alignfalse/v2').encode()).hexdigest()
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / (key + '.npz')
        device = next(encoder.parameters()).device
        if path.exists() and path.with_suffix('.json').exists():
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: torch.tensor(z[k], device=device) for k in ('points', 'features', 'valid', 'view_directions')}
            manifest = json.loads(path.with_suffix('.json').read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
                raise ValueError('CAD appearance cache hash mismatch')
            return cls(**arrays, manifest=manifest)
        begun = time.monotonic()
        if callable(appearance):
            appearance = appearance()
        points = torch.tensor(mesh['points'][::2], device=device)
        diameter = float(mesh['diameter'])
        k = torch.tensor([[390., 0, 111.5], [0, 390., 111.5], [0, 0, 1]], device=device)
        rotations = Rotation.random(12, random_state=np.random.default_rng(42)).as_matrix()
        rgbs, coordinates, validity, directions = [], [], [], []
        for r in rotations:
            pose = torch.eye(4, device=device)
            pose[:3, :3] = torch.tensor(r, device=device, dtype=torch.float32)
            pose[2, 3] = 2 * diameter
            rendered = renderer(appearance, pose, k, 224)
            xy, _, depth = project_and_jacobian(pose, points, k)
            surface = sample_pixels(rendered['depth'], xy)[:, 0]
            visible = (surface > 0) & ((surface - depth).abs() < max(.002, .01 * diameter))
            visible &= ((xy >= 0) & (xy < 224)).all(-1)
            rgbs.append(rendered['rgb'])
            coordinates.append(xy)
            validity.append(visible)
            directions.append(F.normalize(-(pose[:3, :3].T @ pose[:3, 3]), dim=0))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, features = encoder(normalize_rgb(torch.stack(rgbs)))
        descriptors = sample_features(features.float(), torch.stack(coordinates))
        arrays = dict(points=points, features=descriptors, valid=torch.stack(validity), view_directions=torch.stack(directions))
        temporary = path.with_name(path.stem + f'.{os.getpid()}.tmp.npz')
        np.savez(temporary, **{k: v.cpu().numpy() for k, v in arrays.items()})
        manifest = dict(cache_key=key, encoder_sha256=encoder.checkpoint_sha256, mesh_hash=mesh_hash,
            views=12, points=len(points), initialization_dino_images=12, sequence_image_reads=0,
            source='textured static CAD; optional rigid branch only', elapsed_s=time.monotonic() - begun,
            sha256=hashlib.sha256(temporary.read_bytes()).hexdigest())
        temporary.replace(path)
        metadata = path.with_suffix(f'.{os.getpid()}.tmp.json')
        metadata.write_text(json.dumps(manifest, indent=2))
        metadata.replace(path.with_suffix('.json'))
        return cls(**arrays, manifest=manifest)
