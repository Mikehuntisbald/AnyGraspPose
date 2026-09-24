"""Deployment-accessible chronological inputs paired with separate training labels."""
from dataclasses import dataclass
import numpy as np
import torch
import cv2
from lip.data.jepa_sequences import EpisodeFactory
from lip.geometry.so3 import center_pose
from lip.geometry.appearance_renderer import AppearanceRenderer, load_appearance_mesh
from lip.jepa.cad_anchors import CADAnchorBank, cad_asset_hash


@dataclass(frozen=True)
class RigidEpisodeInputs:
    rgb: torch.Tensor
    depth_m: torch.Tensor
    initial_pose_centered: torch.Tensor
    mesh: dict
    intrinsics: torch.Tensor
    timestamps_s: torch.Tensor
    stream_id: str
    bank: CADAnchorBank


@dataclass(frozen=True)
class RigidEpisodeTargets:
    poses_centered: torch.Tensor
    visible_masks: torch.Tensor
    provenance: dict


def sample_center_perturbation(rng, diameter):
    """Bounded displacement of object center and independent rotation about center."""
    tier = int(rng.choice(3, p=[.7, .25, .05]))
    dlo, dhi, alo, ahi = [(0., .02, 0., 5.), (.02, .05, 5., 15.), (.05, .1, 15., 30.)][tier]
    direction, axis = rng.normal(size=3), rng.normal(size=3)
    magnitude, angle = rng.uniform(dlo, dhi), rng.uniform(alo, ahi)
    displacement = direction / np.linalg.norm(direction) * diameter * magnitude
    rotvec = axis / np.linalg.norm(axis) * np.deg2rad(angle)
    return displacement, rotvec, dict(tier=tier, center_displacement_d=magnitude, rotation_deg=angle,
        convention='independent object-center translation and rotation; applied to legal native initializer')


class RigidEpisodeFactory(EpisodeFactory):
    def __init__(self, data_root, index_root, encoder, split='train', device='cuda', packed_root=None):
        super().__init__(data_root, index_root, split, device)
        self.encoder = encoder
        self.renderer = AppearanceRenderer(device)
        self.banks = {}
        self.pose_labels = {}
        self.packed = None
        if packed_root is not None:
            if split != 'train':
                raise ValueError('Packed input cache is restricted to training')
            from lip.data.jepa_rigid_cache import RigidFrameCache
            self.packed = RigidFrameCache(packed_root, self.audit['split_hash'], self.initializers_sha256,
                {sid for sid, _ in self.options})

    def mesh(self, stream, texture=False):
        mesh = super().mesh(stream, texture)
        if not isinstance(mesh['vertices'], torch.Tensor):
            mesh['vertices'] = torch.as_tensor(mesh['vertices'], device=self.device)
            mesh['faces'] = torch.as_tensor(mesh['faces'], device=self.device, dtype=torch.int32)
        return mesh

    def prepare_cpu(self, seed, supervised=8, burn=32):
        rng = np.random.default_rng(seed)
        sid, initial = self.options[int(rng.integers(len(self.options)))]
        stream = self.streams[sid]
        first = int(initial['frame_index'])
        count = burn + supervised
        if first + count > stream['num_frames']:
            raise ValueError('Rigid episode crosses sequence boundary')
        oid = stream['object_id']
        directory = self.root / stream['relative_dir']
        packed_entries = self.packed.stream(sid) if self.packed else None
        def decode(frame):
            if packed_entries is not None:
                return self.packed.decode(packed_entries, frame, oid)
            rgb = cv2.imread(str(directory / f'color_{frame:06d}.jpg'))
            depth = cv2.imread(str(directory / f'aligned_depth_to_color_{frame:06d}.png'), -1)
            if rgb is None or depth is None:
                raise FileNotFoundError(directory)
            with np.load(directory / f'labels_{frame:06d}.npz', allow_pickle=False) as labels:
                mask = (labels['seg'] == oid).copy()
            return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1), depth[None], mask[None]
        decoded = list(self.pool.map(decode, range(first, first + count)))
        if sid not in self.pose_labels:
            with np.load(self.index / stream['pose_cache'], allow_pickle=False) as z:
                self.pose_labels[sid] = z['poses'].copy()
        arrays = (np.stack([x[0] for x in decoded]), np.stack([x[1] for x in decoded]).astype('f4'),
                  np.stack([x[2] for x in decoded]), self.pose_labels[sid][first:first+count])
        return sid, initial, first, count, rng, arrays

    def upload(self, array):
        return torch.tensor(array, device=self.device)

    def sample_rigid(self, seed, supervised=8, burn=32, perturb=True):
        from scipy.spatial.transform import Rotation
        sid, initial, first, count, rng, arrays = self.prepare_cpu(seed, supervised, burn)
        stream = self.streams[sid]
        mesh, oid = self.mesh(stream), stream['object_id']
        if oid not in self.banks:
            appearance = lambda: load_appearance_mesh(self.root / stream['mesh_path'], mesh['center'], self.device)
            self.banks[oid] = CADAnchorBank.create(mesh, appearance, self.renderer, self.encoder,
                'cache/cad_anchors', cad_asset_hash(self.root / stream['mesh_path']))
        rgb = self.upload(arrays[0]).float() / 255
        depth = self.upload(arrays[1]) * self.audit['depth_scale_to_m']
        masks = self.upload(arrays[2])
        center = torch.tensor(mesh['center'], device=self.device)
        truth = center_pose(self.upload(arrays[3]), center)
        pose = center_pose(torch.tensor(initial['pose_original'], device=self.device), center)
        perturbation = 'none'
        if perturb and rng.uniform() < .3:
            displacement, rotvec, perturbation = sample_center_perturbation(rng, float(mesh['diameter']))
            pose = pose.clone()
            pose[:3, :3] = torch.tensor(Rotation.from_rotvec(rotvec).as_matrix(), device=self.device, dtype=torch.float32) @ pose[:3, :3]
            pose[:3, 3] += torch.tensor(displacement, device=self.device, dtype=torch.float32)
        inputs = RigidEpisodeInputs(rgb, depth, pose, mesh, torch.tensor(stream['intrinsics'], device=self.device),
            torch.arange(first, first + count, dtype=torch.float64, device=self.device) / 30., sid + f'|episode{seed}', self.banks[oid])
        targets = RigidEpisodeTargets(truth, masks, dict(seed=seed, stream_id=sid, physical_sequence='/'.join(sid.split('/')[:2]),
            first_frame=first, burn_in=burn, supervised_frames=supervised, perturbation=perturbation,
            initialization='native PoseCNN with explicitly sampled training-only perturbation',
            student_current_gt_pose=False, gt_masks_as_input=False))
        return inputs, targets
