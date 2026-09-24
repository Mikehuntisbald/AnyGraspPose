"""Immutable native bytes; never precomputed pose-conditioned student features."""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import tarfile
import cv2
import numpy as np


class RigidFrameCache:
    def __init__(self, root, split_hash, initializers_sha256, expected_streams, allow_subset=False):
        self.root = Path(root)
        content = (self.root / 'manifest.json').read_bytes()
        self.manifest_sha256 = hashlib.sha256(content).hexdigest()
        manifest = json.loads(content)
        if manifest['split'] != 'train' or manifest['split_hash'] != split_hash or manifest['initializers_sha256'] != initializers_sha256:
            raise ValueError('Packed frame cache provenance mismatch')
        if not allow_subset and (not manifest['completed'] or manifest['subset']):
            raise ValueError('Formal C requires a complete training cache')
        self.records = {r['stream_id']: r for r in manifest['records']}
        if not allow_subset and set(self.records) != set(expected_streams):
            raise ValueError('Packed frame cache population differs from legal native episodes')
        self.loaded = OrderedDict()

    def stream(self, sid):
        if sid in self.loaded:
            self.loaded.move_to_end(sid)
            return self.loaded[sid]
        record = self.records[sid]
        data = (self.root / record['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Packed native frame archive hash mismatch')
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
            entries = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
        if hashlib.sha256(entries['source_manifest.json']).hexdigest() != record['source_manifest_sha256']:
            raise ValueError('Packed source manifest hash mismatch')
        self.loaded[sid] = entries
        while len(self.loaded) > 32:
            self.loaded.popitem(last=False)
        return entries

    @staticmethod
    def decode(entries, frame, object_id):
        rgb = cv2.imdecode(np.frombuffer(entries[f'color_{frame:06d}.jpg'], np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(entries[f'aligned_depth_to_color_{frame:06d}.png'], np.uint8), cv2.IMREAD_UNCHANGED)
        with np.load(io.BytesIO(entries[f'labels_{frame:06d}.npz']), allow_pickle=False) as labels:
            mask = (labels['seg'] == object_id).copy()
        if rgb is None or depth is None:
            raise ValueError('Native byte decode failed')
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1), depth[None], mask[None]
