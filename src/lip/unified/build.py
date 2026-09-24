"""Explicit model/data/software identity for the unified experiment."""
from pathlib import Path
import torch
from lip.jepa.encoder import FrozenDINO
from lip.engine.jepa_checkpoint import sha
from .utonia import FrozenUtonia
from .model import UnifiedTracker,configure_parameters
from .cad import CADStore


def build_model(config,device='cuda'):
    if config['architecture_id'] in ('stream_two_input_jepa_v9', 'stream_conv_cross_jepa_v10','stream_conv_cross_geohistory_jepa_v10','stream_conv_cross_supported_history_jepa_v10','stream_conv_cross_dense_history_jepa_v10','stream_recovered_relation_jepa_v11','stream_cad_surface_jepa_v12'):
        from .two_stream import TwoStreamTracker
        from .conv_cross import ConvCrossTracker
        from .geometric_history import GeometricHistoryTracker
        from .supported_history import SupportedHistoryTracker
        from .dense_history import DenseHistoryTracker
        from .recovered_relation import RecoveredRelationTracker
        from .cad_surface import CADSurfaceTracker
        from .fp_encoder import CachedUtonia
        if set(config['weights'])!={'dino_checkpoint','utonia_checkpoint'} or any(config['paths'].get(k) for k in ('parent','initial_checkpoint','fp_checkpoint')):
            raise ValueError('Clean two-stream initialization forbids previous experiment/FP weights')
        p=config['paths']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'])
            encoder=FrozenDINO(p['dino_repo'],p['dino_checkpoint'],config['weights']['dino_checkpoint'],
                feature_layers=config.get('dino_layers',{}).get('student',(6,12)))
            static=CachedUtonia(p['utonia_checkpoint'],config['weights']['utonia_checkpoint'])
            tracker = {x.architecture_id:x for x in (TwoStreamTracker,ConvCrossTracker,GeometricHistoryTracker,SupportedHistoryTracker,DenseHistoryTracker,RecoveredRelationTracker,CADSurfaceTracker)}[config['architecture_id']]
            model=tracker(encoder,static).to(device)
        if isinstance(model,CADSurfaceTracker):
            import json
            metadata=Path(config['cad_surface']['models_info'])
            if sha(metadata)!=config['cad_surface']['models_info_sha256']:raise ValueError('CAD symmetry metadata hash mismatch')
            info=json.loads(metadata.read_text());identities={}
            for row in (Path(p['index_root'])/'streams.jsonl').read_text().splitlines():
                stream=json.loads(row);name=Path(stream['mesh_path']).parent.name;oid=str(stream['object_id'])
                if name in identities and identities[name]!=oid:raise ValueError('Conflicting CAD object identity')
                identities[name]=oid
            model.cad_surface_symmetry={name:info[oid] for name,oid in identities.items()}
            model.cad_surface_cache=config['cad_surface']['cache']
            model.cad_surface_count=config['cad_surface']['tokens']
        configure_parameters(model)
        if config.get("ema_encoder", {}).get("enabled"):
            from .ema_encoder import attach_ema
            attach_ema(model, config)
        if config.get('cad_rope3d',{}).get('enabled',False):
            from .rope3d import enable_cad_rope3d
            enable_cad_rope3d(model,config['cad_rope3d'])
        model.disable_history=config['runtime'].get('disable_history',False)
        for block in model.core.blocks:block.disable_history=model.disable_history
        model.enable_compilation(config['runtime'].get('compile_frame',False))
        if config["runtime"].get("compile_dino", False):
            from .encoder_performance import compile_encoders
            compile_encoders(model)
        return model
    if config['architecture_id']=='stream_dino_fp_staticutonia_jepa_rgbd_v3':
        return build_fp_model(config,device)
    if config['architecture_id']!=UnifiedTracker.architecture_id:
        raise ValueError('Architecture version mismatch; use the preserved v1 runtime for v1 experiments')
    p=config['paths']
    for key,digest in config['weights'].items():
        if sha(p[key])!=digest:raise ValueError('Weight identity mismatch: '+key)
    encoder=FrozenDINO(p['dino_repo'],p['dino_checkpoint'],config['weights']['dino_checkpoint']).to(device)
    point=FrozenUtonia(p['utonia_checkpoint'],config['weights']['utonia_checkpoint'],
                      fast=config['runtime'].get('utonia_fast',False)).to(device)
    parent=torch.load(p['parent'],map_location='cpu',weights_only=False)
    model=UnifiedTracker(encoder,point,parent['model']).to(device)
    model.observation_points=int(config['runtime'].get('observation_points',4096))
    if model.observation_points not in (1024,2048,4096):raise ValueError('Unsupported observation point budget')
    configure_parameters(model)
    return model


def make_store(config,model):return CADStore(config['paths']['cad_cache'],model.utonia,next(model.parameters()).device)


def build_fp_model(config,device='cuda'):
    from .fp_encoder import CachedUtonia,FrozenFPEncoder
    from .fp_model import FPUnifiedTracker
    p=config['paths']
    for key,digest in config['weights'].items():
        if sha(p[key])!=digest:raise ValueError('Weight identity mismatch: '+key)
    encoder=FrozenDINO(p['dino_repo'],p['dino_checkpoint'],config['weights']['dino_checkpoint']).to(device)
    static=CachedUtonia(p['utonia_checkpoint'],config['weights']['utonia_checkpoint'])
    fp=FrozenFPEncoder(p['fp_root'],p['fp_checkpoint'],config['weights']['fp_checkpoint'],config['runtime'].get('fp_size',160)).to(device)
    fp.to(memory_format=torch.channels_last)
    parent=torch.load(p['parent'],map_location='cpu',weights_only=False)
    model=FPUnifiedTracker(encoder,static,parent['model'],fp).to(device)
    if p.get('initial_checkpoint'):
        record=torch.load(p['initial_checkpoint'],map_location='cpu',weights_only=False)
        if record['architecture_id']!='stream_dino_utonia_jepa_rgbd_v2':raise ValueError('Expected explicit v2 migration source')
        result=model.load_state_dict(record['model'],strict=False)
        missing=[k for k in result.missing_keys if not k.startswith('encoder.') and '.encoder.' not in k and not k.startswith(('fp_observation_proj.','fp_pair_proj.'))]
        if missing or result.unexpected_keys:raise ValueError(f'Unexpected architecture migration keys: {missing}, {result.unexpected_keys}')
        model.migration=dict(checkpoint_sha256=sha(p['initial_checkpoint']),source_step=record['step'],
                             optimizer_reset=True,memory_reset=True,new_modules=['fp_observation_proj','fp_pair_proj'])
    configure_parameters(model)
    if config['runtime'].get('pose_relation_fusion',False):model.enable_pose_relation()
    model.pose_patch_only=config['runtime'].get('pose_patch_only',False)
    if model.pose_patch_only and model.pose_relation_fusion:raise ValueError('Patch-only pose cannot use an FP readout bypass')
    if config['runtime'].get('jepa_pose_geometry',False):model.enable_jepa_pose_geometry()
    model.pair_into_patch=config['runtime'].get('pair_into_patch',False)
    if model.pair_into_patch and not model.pose_patch_only:raise ValueError('Spatial comparison input requires patch-only pose readout')
    model.pose_final_patch_only=config['runtime'].get('pose_final_patch_only',False)
    if model.pose_final_patch_only and not model.pose_patch_only:raise ValueError('Final-patch readout cannot accept direct state/context inputs')
    if config['runtime'].get('fp_observation_only',False):
        if not model.pose_final_patch_only or not model.jepa_pose_geometry:
            raise ValueError('Observation-only FP requires final-patch readout and explicit JEPA geometry')
        model.disable_pair_features()
    model.enable_compilation(config['runtime'].get('compile_frame',False))
    return model
