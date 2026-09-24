"""Causal real episodes and continuous synthetic rigid trajectories.

Real input crops use only an accepted initial native PoseCNN pose. Current labels
are read separately. No synthetic sequence claims measured physical dynamics.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from lip.jepa.contracts import FramePacket, TrainTargets
from lip.data.jepa_pairs import normalize_rgb, add_occlusion


@dataclass
class Episode:
    packets: list[FramePacket]
    teacher_rgb: torch.Tensor
    target_valid: torch.Tensor
    added_fraction: torch.Tensor
    visible_target: torch.Tensor
    support_target: torch.Tensor
    burn_in: int
    provenance: dict
    clean_all_rgb: torch.Tensor | None = None
    support_all: torch.Tensor | None = None


def collate_episodes(episodes):
    """Batch independent lanes with equal burn-in/unroll; no sequence concatenation."""
    from dataclasses import replace
    burn = episodes[0].burn_in
    count = len(episodes[0].packets)
    if any(e.burn_in != burn or len(e.packets) != count for e in episodes):
        raise ValueError('Bucket episodes by chronological length before batching')
    packets = []
    for t in range(count):
        source = [e.packets[t] for e in episodes]
        fields = {name: torch.cat([getattr(p, name) for p in source]) for name in
            ('rgb_crop', 'pixel_valid', 'A_image_to_crop', 'timestamp_s', 'track_prompt', 'source_size_wh')}
        packets.append(replace(source[0], **fields, stream_id=tuple(p.stream_id[0] for p in source)))
    fields = {name: torch.stack([getattr(e, name) for e in episodes], dim=1).flatten(0, 1) for name in
        ('teacher_rgb', 'target_valid', 'added_fraction', 'visible_target', 'support_target')}
    return Episode(packets=packets, burn_in=burn, provenance=dict(lanes=[e.provenance for e in episodes]), **fields)


class EpisodeFactory:
    def __init__(self, data_root, index_root, split='train', device='cuda'):
        cv2.setNumThreads(0)
        self.root, self.index = Path(data_root), Path(index_root)
        self.split, self.device = split, device
        self.audit = json.loads((self.index / 'audit.json').read_text())
        streams = [json.loads(l) for l in (self.index / 'streams.jsonl').read_text().splitlines() if json.loads(l)['split'] == split]
        self.streams = {s['stream_id']: s for s in streams}
        path = Path(f'/mnt/why/dexycb_lip/posecnn_{split}_20260915/runs/full/initializers.json')
        initializers = json.loads(path.read_text())
        if initializers['split_hash'] != self.audit['split_hash'] or initializers['uses_gt_pose']:
            raise ValueError('Invalid episode initialization provenance')
        self.options = []
        self.uninitialized = []
        self.insufficient_history = []
        for key, item in initializers['initializers'].items():
            sid = key.split('|')[0]
            if sid not in self.streams:
                continue
            if item is None:
                self.uninitialized.append(sid)
            elif self.streams[sid]['num_frames'] - item['frame_index'] >= 40:
                self.options.append((sid, item))
            else:
                self.insufficient_history.append(sid)
        self.options.sort(key=lambda x: (x[0], x[1]['frame_index']))
        if not self.options:
            raise ValueError('No legal 40-frame chronological episodes')
        self.pool = ThreadPoolExecutor(4)
        self.meshes = {}
        self.renderer = None
        self.initializers_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def mesh(self, stream, texture=False):
        oid = stream['object_id']
        if oid not in self.meshes:
            with np.load(self.index / stream['mesh_cache']) as z:
                self.meshes[oid] = {k: z[k].copy() for k in z.files}
        mesh = self.meshes[oid]
        if texture and 'appearance' not in mesh:
            from lip.geometry.appearance_renderer import AppearanceRenderer, load_appearance_mesh
            mesh['appearance'] = load_appearance_mesh(self.root / stream['mesh_path'], mesh['center'], self.device)
            if self.renderer is None:
                self.renderer = AppearanceRenderer(self.device)
        return mesh

    def sample(self, seed, synthetic=None, option_index=None, supervised=8, force_burn=None):
        rng = np.random.default_rng(seed)
        if synthetic is None:
            synthetic = seed % 10 < 7
        extended = synthetic and rng.uniform() < 2 / 7  # 20% of overall 70/30 mixture
        burn = (96 if extended else 32) if force_burn is None else force_burn
        if not synthetic and burn != 32:
            raise ValueError('Real release cannot provide extended burn-in')
        stride = int(rng.choice([1, 2, 4])) if synthetic else 1
        count = burn + supervised
        option_index = int(rng.integers(len(self.options))) if option_index is None else option_index
        sid, initial = self.options[option_index]
        stream = self.streams[sid]
        first = int(initial['frame_index'])
        directory = self.root / stream['relative_dir']
        from lip.geometry.crop import crop_matrix, crop_images
        from lip.geometry.so3 import center_pose
        if synthetic:
            from scipy.spatial.transform import Rotation
            mesh = self.mesh(stream, texture=True)
            d = float(mesh['diameter'])
            base_rgb = cv2.imread(str(directory / f'color_{first:06d}.jpg'))
            if base_rgb is None:
                raise FileNotFoundError(directory)
            background = torch.tensor(cv2.cvtColor(base_rgb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy(), device=self.device).float() / 255
            background = F.interpolate(background[None], (224, 224), mode='bilinear', align_corners=False)[0]
            initial_rotation = Rotation.random(random_state=rng).as_matrix()
            angular_velocity = rng.normal(0, .15, 3)
            phase = rng.uniform(0, 6.28)
            k = torch.tensor([[390., 0, 111.5], [0, 390., 111.5], [0, 0, 1]], device=self.device)
            clean, support = [], []
            for t in range(count):
                seconds = t * stride / 30.
                pose = torch.eye(4, device=self.device)
                pose[:3, :3] = torch.tensor(Rotation.from_rotvec(angular_velocity * seconds).as_matrix() @ initial_rotation, device=self.device, dtype=torch.float32)
                pose[:3, 3] = torch.tensor([.12 * d * np.sin(seconds + phase), .08 * d * np.cos(.7 * seconds + phase), 2 * d], device=self.device)
                render = self.renderer(mesh['appearance'], pose, k, 224)
                clean.append(torch.where(render['mask'][None], render['rgb'], background))
                support.append(render['mask'][None].float())
            clean, support = torch.stack(clean), torch.stack(support)
            affine = torch.eye(3, device=self.device)
            pixel_valid = torch.ones(count, 1, 224, 224, device=self.device, dtype=torch.bool)
            size_wh = [224., 224.]
            times = np.arange(count) * stride / 30.
        else:
            if first + count > stream['num_frames']:
                raise ValueError('Episode would cross sequence boundary')
            mesh = self.mesh(stream)
            base = center_pose(torch.tensor(initial['pose_original'], device=self.device), torch.tensor(mesh['center'], device=self.device))
            k = torch.tensor(stream['intrinsics'], device=self.device)
            affine, _ = crop_matrix(torch.tensor(mesh['vertices'], device=self.device), base, k, 224, 1.4)
            def decode(frame):
                image = cv2.imread(str(directory / f'color_{frame:06d}.jpg'))
                if image is None:
                    raise FileNotFoundError(directory / f'color_{frame:06d}.jpg')
                with np.load(directory / f'labels_{frame:06d}.npz', allow_pickle=False) as z:
                    target = (z['seg'] == stream['object_id']).copy()
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1), target[None]
            decoded = list(self.pool.map(decode, range(first, first + count)))
            images = torch.tensor(np.stack([v[0] for v in decoded]), device=self.device).float() / 255
            masks = torch.tensor(np.stack([v[1] for v in decoded]), device=self.device).float()
            clean = crop_images(images, affine)
            support = crop_images(masks, affine, mode='nearest')
            pixel_valid = crop_images(torch.ones_like(images[:, :1]), affine) > .999
            size_wh = [640., 480.]
            times = np.arange(first, first + count) / 30.
        corrupted, occluded = [], []
        duration = int(rng.choice([8, 16, 32, 64]))
        recovery = bool(rng.integers(2))
        onset = max(8, burn - duration + 4 if recovery else burn - duration // 2)
        # One fixed shape/color follows a continuous path; early anchor is genuinely observed.
        for t in range(count):
            if t < onset or t >= onset + duration:
                image, occ = clean[t:t+1], torch.zeros_like(support[t:t+1], dtype=torch.bool)
            else:
                progress = (t - onset) / duration
                image, occ = add_occlusion(clean[t:t+1], [seed + 6143], time=float(times[t]), motion_offset_x=-400 + 800 * progress)
            corrupted.append(image)
            occluded.append(occ)
        student, occ = torch.cat(corrupted), torch.cat(occluded)
        normalized = normalize_rgb(student)
        identity = f'{sid}|episode{seed}|synthetic{int(synthetic)}'
        packets = [FramePacket(normalized[t:t+1], pixel_valid[t:t+1], affine[None],
            torch.tensor([times[t]], dtype=torch.float64, device=self.device), (identity,),
            torch.tensor([[32., 32., 192., 192.]], device=self.device), torch.tensor([size_wh], device=self.device)) for t in range(count)]
        fraction = F.avg_pool2d(support[burn:], 14, 14).flatten(1)
        bounds = F.avg_pool2d(pixel_valid[burn:].float(), 14, 14).flatten(1) >= .999
        added = F.avg_pool2d(occ[burn:].float(), 14, 14).flatten(1)
        visible = F.avg_pool2d(support[burn:] * ~occ[burn:], 14, 14).flatten(1).masked_fill(~bounds, float('nan'))
        amodal = fraction if synthetic else torch.full_like(fraction, float('nan'))
        return Episode(packets, normalize_rgb(clean[burn:]), (fraction >= .9) & bounds, added, visible, amodal, burn,
            dict(seed=seed, kind='synthetic_continuous' if synthetic else 'real_added_occlusion', physical_sequence='/'.join(sid.split('/')[:2]),
                stream_id=sid, first_frame=first, stride=stride, burn_in=burn, supervised_frames=supervised,
                occluder_duration_frames=duration, occluder_onset_frame=onset, recovery_episode=recovery,
                timestamp_source='synthetic simulation clock' if synthetic else 'frame index / nominal 30 Hz',
                crop_source='synthetic fixed camera' if synthetic else 'first native PoseCNN initializer, fixed thereafter',
                gt_inputs=False, dynamics_are_physical_ground_truth=False if synthetic else None),
            clean_all_rgb=clean, support_all=support)

    def targets(self, episode, encoder):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            mid, last = encoder(episode.teacher_rgb)
        return TrainTargets(mid.float(), last.float(), episode.target_valid, episode.added_fraction,
                            episode.visible_target, episode.support_target, episode.provenance)


def read_soft_manifest(path):
    """Require chronological CAD-free data; never auto-promote warp smoke to real soft."""
    manifest = json.loads(Path(path).read_text())
    for record in manifest['sequences']:
        required = ('rgb_paths', 'timestamps', 'point_tracks_xy', 'point_visible', 'point_valid')
        if any(k not in record for k in required):
            raise ValueError('Incomplete soft tracking record')
        if 'first_bbox' not in record and 'first_mask' not in record:
            raise ValueError('Soft initialization requires a first bbox or mask')
        if not np.all(np.diff(record['timestamps']) > 0):
            raise ValueError('Soft timestamps must be chronological')
        if len(record['rgb_paths']) != len(record['timestamps']):
            raise ValueError('Soft frame/time length mismatch')
    return manifest


class EpisodeCache:
    """Clean scene cache with fresh continuous RGB corruption on every sampled episode."""
    def __init__(self, root, encoder_sha256):
        self.root = Path(root)
        manifests = sorted(self.root.glob('rank*/manifest.json'))
        if len(manifests) != 8:
            raise ValueError('Expected all eight temporal cache shards')
        self.records = []
        self.manifest_hashes = []
        for path in manifests:
            m = json.loads(path.read_text())
            if not m['completed'] or m['encoder_sha256'] != encoder_sha256 or m['split'] != 'train':
                raise ValueError('Unsealed or stale temporal cache')
            self.manifest_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            self.records.extend(dict(r, path=str(path.parent / r['file'])) for r in m['records'])
        self.buckets = {burn: [r for r in self.records if r['burn_in'] == burn] for burn in (32, 96)}
        if any(not v for v in self.buckets.values()):
            raise ValueError('Temporal cache lacks a burn-in bucket')

    def sample(self, seed, burn, device='cuda'):
        rng = np.random.default_rng(seed)
        record = self.buckets[burn][int(rng.integers(len(self.buckets[burn])))]
        with np.load(record['path'], allow_pickle=False) as z:
            arrays = {k: torch.from_numpy(z[k].copy()).to(device) for k in z.files}
        clean, support = arrays['rgb'].float() / 255, arrays['support'].float() / 255
        count = len(clean)
        duration = int(rng.choice([8, 16, 32, 64]))
        recovery = bool(rng.integers(2))
        onset = max(8, burn - duration + 4 if recovery else burn - duration // 2)
        students, occs = [], []
        times = arrays['times'].double()
        for t in range(count):
            if t < onset or t >= onset + duration:
                image, occ = clean[t:t+1], torch.zeros_like(support[t:t+1], dtype=torch.bool)
            else:
                image, occ = add_occlusion(clean[t:t+1], [seed + 6143], time=float(times[t]), motion_offset_x=-400 + 800 * (t - onset) / duration)
            students.append(image)
            occs.append(occ)
        students, occ = normalize_rgb(torch.cat(students)), torch.cat(occs)
        packets = [FramePacket(students[t:t+1], arrays['pixel_valid'][t:t+1].bool(), arrays['affine'][None].float(),
            times[t:t+1], (record['stream_id'] + f'|draw{seed}',), arrays['prompt'][None].float(), arrays['size_wh'][None].float()) for t in range(count)]
        fraction = F.avg_pool2d(support[burn:], 14, 14).flatten(1)
        bounds = F.avg_pool2d(arrays['pixel_valid'][burn:].float(), 14, 14).flatten(1) >= .999
        added = F.avg_pool2d(occ[burn:].float(), 14, 14).flatten(1)
        visible = F.avg_pool2d(support[burn:] * ~occ[burn:], 14, 14).flatten(1).masked_fill(~bounds, float('nan'))
        amodal = fraction if record['kind'] == 'synthetic_continuous' else torch.full_like(fraction, float('nan'))
        provenance = dict(record, draw_seed=seed, occluder_duration_frames=duration, occluder_onset_frame=onset, recovery_episode=recovery)
        episode = Episode(packets, normalize_rgb(clean[burn:]), (fraction >= .9) & bounds, added, visible, amodal, burn, provenance)
        targets = TrainTargets(arrays['features'][:, 0].float(), arrays['features'][:, 1].float(),
            episode.target_valid, added, visible, amodal, provenance)
        return episode, targets


def collate_targets(targets):
    return TrainTargets(**{name: torch.stack([getattr(t, name) for t in targets], dim=1).flatten(0, 1) for name in
        ('teacher_features_mid', 'teacher_features_last', 'feature_target_valid', 'added_occlusion_fraction',
         'object_visible_target', 'object_support_target')}, provenance=dict(lanes=[t.provenance for t in targets]))
