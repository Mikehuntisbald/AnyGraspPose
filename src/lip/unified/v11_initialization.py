"""Explicit V10 -> V11 weights-only migration with a fresh optimizer phase."""
import torch
from lip.engine.jepa_checkpoint import sha, core_state
from lip.jepa.config import config_hash
from .checkpoint import software_environment
from .recovered_relation import initialize_from_supported


def initialize_training(model, config, world, provenance=None):
    plan = config['v11_initialization']
    path = plan['source_checkpoint']
    if sha(path) != plan['source_sha256']:
        raise ValueError('V11 source checkpoint hash mismatch')
    source = torch.load(path, map_location='cpu', weights_only=False)
    if source['architecture_id'] != 'stream_conv_cross_supported_history_jepa_v10' or model.architecture_id != 'stream_recovered_relation_jepa_v11':
        raise ValueError('Unexpected V11 initialization architectures')
    if source['config_hash'] != config_hash(source['config']) or source['software_environment'] != software_environment():
        raise ValueError('V11 source configuration/environment mismatch')
    if source['config']['weights'] != config['weights']:
        raise ValueError('Frozen encoder mismatch')
    step = plan['source_step']
    if config['migration']['source_step'] != 0:
        raise ValueError('V11 stage must start at zero')
    if source['step'] != step or source['sampler_position'] != step * config['training']['effective_batch'] or len(source['rng']) != world:
        raise ValueError('V11 source step/sampler/world mismatch')
    initialize_from_supported(model, source['model'])
    current = core_state(model)
    if not all(torch.equal(value.cpu(), current[name].cpu()) for name, value in source['model'].items()):
        raise ValueError('V10 core tensors changed during initialization')
    receipt = dict(kind='v10_to_v11_weights_only',source_sha256=plan['source_sha256'],source_step=step,
        stage_start=0,rng_reset=True,parent_core_exact=True,new_module='geometry_readout',optimizer_reset=True,scheduler_reset=True,
        sampler_position=0,parent_sampler_position=source['sampler_position'],all_rank_rng=world,memory_reset=True)
    # Stable metadata is reconstructed on strict resume as well as first start.
    model.migration.update(receipt)
    return source, receipt
