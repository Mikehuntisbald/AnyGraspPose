import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, distance
import trimesh


def mesh_metadata(path, cache, scale=1.):
    path, cache = Path(path), Path(cache)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = hashlib.sha256(f'{digest}:{scale}:512:42'.encode()).hexdigest()
    dest = cache / (key+'.npz'); cache.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        mesh = trimesh.load(path, process=False, force='mesh', skip_materials=True)
        v = np.asarray(mesh.vertices, dtype=np.float64)*scale
        c = (v.max(0)+v.min(0))/2
        hull = v[ConvexHull(v).vertices]
        diameter = max(distance.cdist(hull[i:i+1024], hull).max() for i in range(0, len(hull), 1024))
        mesh.vertices = v-c
        # Explicit RNG: area weighted triangle surface sampling.
        rng = np.random.default_rng(42)
        tris = np.asarray(mesh.triangles)
        areas = np.linalg.norm(np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0]), axis=-1)
        tri = tris[rng.choice(len(tris), 512, p=areas/areas.sum())]
        uv = rng.random((512, 2)); uv[uv.sum(-1)>1] = 1-uv[uv.sum(-1)>1]
        points = tri[:, 0]+uv[:, :1]*(tri[:, 1]-tri[:, 0])+uv[:, 1:]*(tri[:, 2]-tri[:, 0])
        tmp = dest.with_suffix('.tmp.npz')
        np.savez(tmp, vertices=(v-c).astype('f4'), faces=np.asarray(mesh.faces, dtype='i4'),
                 center=c.astype('f4'), diameter=np.float32(diameter), points=points.astype('f4'),
                 source_sha256=digest, cache_key=key, scale_to_m=scale)
        tmp.replace(dest)
    with np.load(dest) as f: return {k:f[k].copy() for k in f.files}, str(dest)
