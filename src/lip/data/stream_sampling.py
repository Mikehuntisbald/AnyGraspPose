"""Load a fixed stream manifest and optionally bind it to a saved config hash."""
import hashlib
import json
from pathlib import Path


def load_fixed_manifest(path, expected_sha256=None):
    if path is None:
        if expected_sha256 is not None:
            raise ValueError('Config requires its fixed sampling manifest')
        return None, dict(mode='online', manifest_sha256=None, entries=None, bound_to_config=False)
    raw=Path(path).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError('Fixed sampling manifest SHA-256 differs from config')
    rows=json.loads(raw)
    if not isinstance(rows,list) or not rows:
        raise ValueError('Fixed sampling manifest must be a nonempty list')
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'stream','start','seed'}:
            raise ValueError('Fixed stream samples require exactly stream, start and seed')
        if any(type(row[key]) is not int or row[key]<0 for key in row):
            raise ValueError('Fixed stream sample indices and noise seed must be nonnegative integers')
    return rows, dict(mode='fixed',manifest_sha256=digest,entries=len(rows),bound_to_config=expected_sha256 is not None)
