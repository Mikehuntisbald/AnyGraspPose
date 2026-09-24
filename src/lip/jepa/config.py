import hashlib
import json
import os
from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    if 'extends' in config:
        base = load_config(path.parent / config.pop('extends'))
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k] = v
        merge(base, config)
        config = base
    def expand(v):
        if isinstance(v, str):
            return os.path.expandvars(v)
        if isinstance(v, dict):
            return {k: expand(x) for k, x in v.items()}
        if isinstance(v, list):
            return [expand(x) for x in v]
        return v
    return expand(config)


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
