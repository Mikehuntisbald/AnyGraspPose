import hashlib
import json
import os
import platform
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from lip.jepa.config import config_hash


def software_environment():
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ('torch', 'numpy', 'xformers', 'nvdiffrast', 'opencv-python', 'PyYAML'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), packages=packages, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), device=torch.cuda.get_device_name(),
        compute_capability=list(torch.cuda.get_device_capability()))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_stage_parent(path, record, stage, variant, seed, split_hash, mesh_hash, encoder_sha256):
    runtime = record['config']['runtime']
    if record.get('stage') != stage or runtime['seed'] != seed or runtime['variant'] != variant:
        raise ValueError('Parent stage/seed/objective identity mismatch')
    for key, value in dict(split_hash=split_hash, mesh_hash=mesh_hash, encoder_sha256=encoder_sha256).items():
        if record['provenance'][key] != value:
            raise ValueError('Parent data/encoder mismatch: ' + key)
    receipt = json.loads((Path(path).parent / 'completion_receipt.json').read_text())
    if not receipt.get('completed') or receipt.get('stage') != stage or receipt.get('optimizer_steps') != 20000:
        raise ValueError('Parent stage has not completed its declared budget')
    digest = sha(path)
    if digest not in (receipt.get('best_sha256'), receipt.get('last_sha256')):
        raise ValueError('Parent checkpoint is not bound to its terminal receipt')
    return digest


def core_state(model):
    if getattr(model, 'trainable_encoder', False):
        return model.state_dict()
    return {k: v for k, v in model.state_dict().items() if not k.startswith('encoder.') and '.encoder.' not in k}


def load_core(model, state, allow_geometry=False):
    if getattr(model, "trainable_encoder", False):
        model.load_state_dict(state, strict=True)
        return
    status = model.load_state_dict(state, strict=False)
    missing = [k for k in status.missing_keys if not k.startswith('encoder.') and '.encoder.' not in k and not (allow_geometry and k.startswith('geometry_adapter.'))]
    if missing or status.unexpected_keys:
        raise ValueError(f'Core state mismatch: {missing}, {status.unexpected_keys}')


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


def save(path, model, optimizer, scheduler, step, config, provenance, best, sampler_position):
    rank = dist.get_rank() if dist.is_initialized() else 0
    states = [rng_state()]
    if dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, rng_state())
    if rank:
        return
    path = Path(path)
    record = dict(architecture_id=model.architecture_id, model_version='dino_jepa_v1.0',
        model=core_state(model), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        global_step=config['runtime'].get('global_step_offset', 0) + step, stage_step=step, stage=config['runtime']['stage'],
        rng=states, config=config, config_hash=config_hash(config), provenance=provenance,
        best_selection_state=best, sampler_position=sampler_position, state_carry_across_updates=False,
        software_environment=software_environment())
    torch.save(record, str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)


def resume(path, model, optimizer, scheduler, config, provenance, rank, world):
    record = torch.load(path, map_location='cpu', weights_only=False)
    for key, expected in [('architecture_id', model.architecture_id), ('config_hash', config_hash(config)), ('provenance', provenance)]:
        if record[key] != expected:
            raise ValueError('Strict resume mismatch: ' + key)
    if len(record['rng']) != world:
        raise ValueError('Resume world size mismatch')
    if record.get('software_environment') != software_environment():
        raise ValueError('Strict resume software environment mismatch')
    load_core(model, record['model'])
    optimizer.load_state_dict(record['optimizer'])
    scheduler.load_state_dict(record['scheduler'])
    restore_rng(record['rng'][rank])
    return record
