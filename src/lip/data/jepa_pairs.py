"""Pixel-space added occlusion; exact clean teacher cache, no token masking shortcut."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lip.jepa.contracts import FramePacket, TrainTargets

MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def normalize_rgb(rgb):
    return (rgb - rgb.new_tensor(MEAN)[None, :, None, None]) / rgb.new_tensor(STD)[None, :, None, None]


def add_occlusion(rgb, seeds, extreme=True, time=None, motion_offset_x=0.):
    """Deterministic spatially connected masks generated before the student encoder.

    time moves a fixed occluder smoothly for episode augmentation. Sampling is
    independent of labels; actual object visibility is measured afterward.
    """
    images, masks = [], []
    for image, seed in zip(rgb, seeds):
        rng = np.random.default_rng(int(seed))
        probabilities = [.25, .35, .30, .10] if extreme else [.28, .39, .33, 0]
        level = int(rng.choice(4, p=probabilities))
        side = rng.uniform(*[(24, 65), (65, 130), (130, 210), (224, 300)][level])
        cx, cy = rng.uniform(50, 174, 2)
        if time is not None:
            cx += 22 * np.sin(float(time) * .7 + rng.uniform(0, 6.28))
            cy += 16 * np.sin(float(time) * .5 + rng.uniform(0, 6.28))
        cx += float(motion_offset_x)
        yy, xx = torch.meshgrid(torch.arange(224, device=image.device), torch.arange(224, device=image.device), indexing='ij')
        angle = rng.uniform(-np.pi, np.pi)
        x = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        y = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        shape = int(rng.integers(3))
        if shape == 0:
            mask = (x.abs() < side / 2) & (y.abs() < side / 2)
        elif shape == 1:
            mask = (x / (side / 2)) ** 2 + (y / (side * .6)) ** 2 < 1
        else:
            mask = (y - 20 * torch.sin(x / 30)).abs() < side / 3
        # Deterministic, genuinely pixel-space textured foreground; no hand labels.
        color = image.new_tensor(rng.uniform(.05, .95, 3))[:, None, None]
        texture = (color + .15 * torch.sin(xx.float()[None] * .17 + yy.float()[None] * .11)).clamp(0, 1)
        images.append(torch.where(mask[None], texture, image))
        masks.append(mask)
    return torch.stack(images), torch.stack(masks)[:, None]


class PairCache:
    def __init__(self, root, split, expected_encoder=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if not self.manifest.get('completed') or self.manifest['split'] != split:
            raise ValueError('Incomplete or wrong-split pair cache')
        if expected_encoder and self.manifest['encoder_sha256'] != expected_encoder:
            raise ValueError('Stale encoder cache')
        self.records = self.manifest['records']
        self.arrays = {name: np.load(self.root / (name + '.npy'), mmap_mode='r') for name in
                       ('rgb', 'support', 'pixel_valid', 'features', 'affine', 'prompt', 'size_wh')}

    def __len__(self):
        return len(self.records)

    def batch(self, indices, seeds, device, extreme=True, unmasked=False):
        idx = np.asarray(indices)
        a = {k: torch.from_numpy(np.array(v[idx], copy=True)).to(device) for k, v in self.arrays.items()}
        rgb = a['rgb'].float() / 255
        student, occ = (rgb, torch.zeros_like(a['support'], dtype=torch.bool)) if unmasked else add_occlusion(rgb, seeds, extreme)
        support = a['support'].float() / 255
        valid_pixels = a['pixel_valid'].bool()
        fraction = F.avg_pool2d(support, 14, 14).flatten(1)
        added = F.avg_pool2d(occ.float(), 14, 14).flatten(1)
        bounds = F.avg_pool2d(valid_pixels.float(), 14, 14).flatten(1) >= .999
        target_valid = (fraction >= .9) & bounds
        visible = F.avg_pool2d(support * ~occ, 14, 14).flatten(1)
        visible = visible.masked_fill(~bounds, float('nan'))
        records = [self.records[int(i)] for i in idx]
        amodal = fraction.clone()
        for i, r in enumerate(records):
            if r['kind'] == 'real_added_occlusion':
                amodal[i] = float('nan')
        packet = FramePacket(normalize_rgb(student), valid_pixels, a['affine'].float(),
            torch.tensor([r['timestamp_s'] for r in records], device=device, dtype=torch.float64),
            tuple(f"{r['stream_id']}|lane{lane}" for lane, r in enumerate(records)), a['prompt'].float(), a['size_wh'].float())
        targets = TrainTargets(a['features'][:, 0].float(), a['features'][:, 1].float(), target_valid, added,
            visible, amodal, dict(kind=[r['kind'] for r in records], indices=idx.tolist(),
                split=self.manifest['split'], pixel_occlusion_before_encoding=True))
        return packet, targets
