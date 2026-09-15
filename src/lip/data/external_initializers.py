"""Train-only real initializer manifest; deterministic mixture and provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np


def initializer_key(stream_id,frame):return stream_id+'|'+str(int(frame))


def request_real_initializer(seed,probability):
    if not 0<=probability<=1:raise ValueError('Invalid real initializer probability')
    return bool(np.random.default_rng(int(seed)^0x41A79D2B).random()<probability)


def load_train_initializers(path,expected_sha,streams,audit):
    if not path or not expected_sha:raise ValueError('Bind an initializer file and SHA')
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Train initializer SHA mismatch')
    manifest=json.loads(raw)
    if not manifest.get('completed') or manifest.get('split')!='train' or manifest.get('uses_gt_pose') is not False or manifest.get('fp_calls')!=0:
        raise ValueError('Completed real train-only zero-FP initializers required')
    if any(manifest.get(k)!=audit[k] for k in ('split_hash','mesh_hash')):raise ValueError('Train initializer data identity mismatch')
    if not manifest.get('checkpoint_sha256') or not manifest.get('inference_source_sha256'):raise ValueError('Real initializer provenance missing')
    registry={s['stream_id']:s for s in streams}
    for key,entry in manifest['initializers'].items():
        sid,frame=key.rsplit('|',1);frame=int(frame)
        if sid not in registry or registry[sid]['split']!='train' or not 0<=frame<registry[sid]['num_frames']:
            raise ValueError('Initializer outside train population')
        if entry is None:continue
        pose=np.asarray(entry['pose_original'],dtype='f4')
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or pose[2,3]<=0 or not np.allclose(pose[3],[0,0,0,1],atol=1e-4):raise ValueError('Illegal real train pose')
        R=pose[:3,:3]
        if not np.allclose(R.T@R,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(R),1.,atol=1e-3):raise ValueError('Illegal real train rotation')
        if entry['stream_id']!=sid or entry['frame_index']!=frame or entry['object_id']!=registry[sid]['object_id'] or not np.isfinite(entry['score']):raise ValueError('Train initializer key mismatch')
    return manifest
