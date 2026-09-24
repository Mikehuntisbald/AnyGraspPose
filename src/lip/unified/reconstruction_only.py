"""Explicit recovery-only continuation; frozen pose modules and crop reference."""
import torch
from .losses import reconstruction_loss, RECONSTRUCTION_METRICS

POSE_PREFIXES = ('head.', 'object_attn.', 'object_norm.', 'geometry_readout.')


def is_pose_parameter(name):
    return name == 'query' or name.startswith(POSE_PREFIXES)


def configure_reconstruction_only(model):
    from .model import configure_parameters
    configure_parameters(model)
    for name, parameter in model.named_parameters():
        if is_pose_parameter(name):
            parameter.requires_grad_(False)
    if getattr(model, "trainable_encoder", False):
        model.encoder.requires_grad_(True)
        model.ema_teacher.requires_grad_(False)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def recovery_objective(output, teacher, truth, points, diameter, weights):
    # Deliberately never inspect pose outputs or call pose_loss, even for logs.
    loss, parts = reconstruction_loss(output, teacher, weights)
    extra = []
    if weights.get('spatial_centered', 0) or weights.get('spatial_correspondence', 0) or weights.get('camera_consistency', 0):
        from .recovery_focus import focus_loss, FOCUS_METRICS
        focus, diagnostics = focus_loss(output, teacher, weights)
        loss = loss + focus
        extra = [diagnostics[key] for key in FOCUS_METRICS]
    if weights.get('cad_surface_correspondence',0):
        from .cad_surface_targets import surface_correspondence_loss
        correspondence,diagnostics=surface_correspondence_loss(output,teacher)
        loss=loss+weights['cad_surface_correspondence']*correspondence
        extra+=list(diagnostics.unbind())
    if weights.get("log_feature_layers",False):
        extra += [parts[key] for key in ("real_feature_mid","real_feature_last","proxy_feature_mid","proxy_feature_last")]
    if weights.get('local_difference', 0) or weights.get('local_correspondence', 0):
        from .local_structure import local_structure_loss, LOCAL_METRICS
        local, diagnostics = local_structure_loss(output, teacher, weights)
        loss = loss + local
        extra += [diagnostics[key] for key in LOCAL_METRICS]
    return loss, torch.stack([loss.detach().new_zeros(()) for _ in range(4)] +
                            [parts[key] for key in RECONSTRUCTION_METRICS] + extra)


def initialize_training(model, config, world):
    from .build import build_model
    from .checkpoint import software_environment
    from lip.engine.jepa_checkpoint import sha, load_core, core_state
    from lip.jepa.config import config_hash
    plan = config['reconstruction_only']
    if model.architecture_id not in ('stream_recovered_relation_jepa_v11','stream_cad_surface_jepa_v12'):
        raise ValueError('Recovery-only continuation expects V11')
    if sha(plan['source_checkpoint']) != plan['source_sha256']:
        raise ValueError('Recovery-only source hash mismatch')
    source = torch.load(plan['source_checkpoint'], map_location='cpu', weights_only=False)
    surface_migration=model.architecture_id=='stream_cad_surface_jepa_v12' and source['architecture_id']=='stream_recovered_relation_jepa_v11'
    if ((source['architecture_id'] != model.architecture_id and not surface_migration) or
            source['step'] != plan['source_step'] or
            source['config_hash'] != config_hash(source['config']) or
            source['software_environment'] != software_environment()):
        raise ValueError('Recovery-only source identity/environment mismatch')
    if (source['sampler_position'] != source['step'] * config['training']['effective_batch'] or
            len(source['rng']) != world or source['config']['weights'] != config['weights'] or
            config['migration']['source_step'] != source['step']):
        raise ValueError('Recovery-only sampler/world/frozen-encoder mismatch')
    if config['runtime'].get('pose_pair_frames') or config['training']['loss_weights'].get('pose_error', 0):
        raise ValueError('Pose objectives are forbidden in recovery-only training')
    if surface_migration:
        from .cad_surface import migrate_surface
        migrate_surface(model,source['model'])
    elif getattr(model, "trainable_encoder", False):
        status = model.load_state_dict(source["model"], strict=False)
        if status.unexpected_keys or any(not k.startswith(("encoder.", "ema_teacher.", "ema_updates")) for k in status.missing_keys):
            raise ValueError("EMA migration state mismatch")
    else:load_core(model, source['model'])
    configure_reconstruction_only(model)
    current = core_state(model)
    if not all(torch.equal(v.cpu(), current[k].cpu()) for k, v in source['model'].items()):
        raise ValueError('Source tensors changed during recovery-only initialization')
    # This fixed model produces training crop/base trajectories only. It is not
    # attached to the student, optimizer, checkpoint, or deployed inference path.
    reference_plan = plan.get('reference', dict(checkpoint=plan['source_checkpoint'], sha256=plan['source_sha256']))
    if sha(reference_plan['checkpoint']) != reference_plan['sha256']:
        raise ValueError('Fixed crop reference hash mismatch')
    reference_source = source if reference_plan['sha256'] == plan['source_sha256'] else torch.load(reference_plan['checkpoint'], map_location='cpu', weights_only=False)
    if reference_source['architecture_id'] not in (model.architecture_id,'stream_recovered_relation_jepa_v11') or reference_source['config']['weights'] != config['weights']:
        raise ValueError('Fixed crop reference architecture/encoder mismatch')
    reference_config=dict(config,architecture_id=reference_source['architecture_id'])
    reference_config.pop('ema_encoder', None)
    reference_config.pop('dino_layers', None)
    reference_config['runtime']=dict(config['runtime'],disable_history=False,compile_dino=False)
    reference = build_model(reference_config)
    load_core(reference, reference_source['model'])
    reference.requires_grad_(False).eval()
    reference.weights_version = 'fixed-training-crop-reference/' + reference_plan['sha256']
    reference.trusted_training_inputs = True
    receipt = dict(kind='jepa_recovery_only', source_sha256=plan['source_sha256'],
                   source_step=source['step'], source_sampler_position=source['sampler_position'],
                   optimizer_reset=True, scheduler_reset=True, rng_reset=False,
                   memory_reset=True, pose_loss_enabled=False, pose_parameters_frozen=True,
                   crop_reference='immutable source V11; same causal student observations; no GT input',
                   teacher_target_max_radius_d=config['supervision']['real_geometry_max_radius_d'])
    receipt['crop_reference_sha256'] = reference_plan['sha256']
    if config.get('dino_layers'):
        receipt.update(dino_layers=config['dino_layers'],ema_source_updates=int(model.ema_updates),
                       readout_initialization='all source tensors retained; feature heads adapt to new target layers')
    model.migration.update(receipt)
    return source, receipt, reference
