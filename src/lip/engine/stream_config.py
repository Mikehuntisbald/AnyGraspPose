import hashlib
import json
import math
from pathlib import Path
import torch
import yaml
from lip.engine.stream_state import cache_contract_for


def load_stream_config(path):
    c=yaml.safe_load(Path(path).read_text())
    if c['architecture_id'] not in ('stream_single','stream_dual','stream_dual_cross','stream_dual_cross_residual','stream_rk_factorial','stream_rk_spatial','stream_rk_aligned','stream_rk_direct_pose','stream_rk_pose_reference','stream_rk_adaptive_reference','stream_rk_rotation_anchor','stream_rk_rotation_anchor_smooth'):raise ValueError('Unknown architecture')
    if c['architecture_id'] in ('stream_dual_cross','stream_dual_cross_residual','stream_rk_factorial','stream_rk_spatial','stream_rk_aligned','stream_rk_direct_pose','stream_rk_pose_reference','stream_rk_adaptive_reference','stream_rk_rotation_anchor','stream_rk_rotation_anchor_smooth') and c.get('cache_kind','functional')!='functional':raise ValueError('Cross cache requires functional mode')
    if c['architecture_id'] in ('stream_rk_factorial','stream_rk_spatial','stream_rk_aligned','stream_rk_direct_pose','stream_rk_pose_reference','stream_rk_adaptive_reference','stream_rk_rotation_anchor','stream_rk_rotation_anchor_smooth'):
        if any(type(c.get(k)) is not bool for k in ('observation_reliability','keyframe_memory')):raise ValueError('Explicit boolean factorial switches required')
        if c['memory_frames']!=8 or c['keyframe_slots']!=4:raise ValueError('Factorial v1 fixes 8 recent frames and 4 anchor slots')
        if c['keyframe_min_gap']<1 or c['keyframe_max_age']<8 or c['support_tolerance']<=0:raise ValueError('Invalid keyframe/support contract')
    if c['architecture_id'] in ('stream_rk_spatial','stream_rk_aligned','stream_rk_direct_pose','stream_rk_pose_reference','stream_rk_adaptive_reference','stream_rk_rotation_anchor','stream_rk_rotation_anchor_smooth') and (not c['observation_reliability'] or not c['keyframe_memory'] or c.get('spatial_memory_side')!=14):raise ValueError('Spatial memory requires R1K1 and a 14x14 encoder grid')
    if c['architecture_id']=='stream_rk_aligned' and (c.get('aligned_query_side')!=14 or c.get('aligned_sigma')!=.05):raise ValueError('Aligned v1 fixes a 14x14 query grid and .05d coordinate bandwidth')
    if c['architecture_id'] in ('stream_rk_adaptive_reference','stream_rk_rotation_anchor','stream_rk_rotation_anchor_smooth') and c.get('reference_write_limit')!=.25:raise ValueError('Adaptive-reference v1 fixes a .25 write limit')
    fixed=dict(source_tokens=17,latent_dim=256,temporal_layers=4,attention_heads=8,image_size=224)
    for k,v in fixed.items():
        if c.get(k)!=v:raise ValueError(f'v2 contract requires {k}={v}')
    if c['memory_frames'] not in (1,8,16):raise ValueError('Use a separately trained W=1,8,16 config')
    if c['precision'] not in ('fp32','bf16'):raise ValueError('Explicit precision required')
    if c.get('basin_weight',0) or c.get('foundationpose_enabled',False) or c.get('depth_correction',False):
        raise ValueError('FP, basin loss and GT-depth correction are not v2 defaults')
    if c['burn_in_frames']<0 or c['supervised_unroll_frames']<1:raise ValueError('Invalid unroll')
    prefix=c.get('startup_supervision_frames',0)
    if type(prefix) is not int or prefix not in (0,8):raise ValueError('Startup supervision must be 0 or 8 frames')
    if prefix and (c['burn_in_frames']!=8 or c['supervised_unroll_frames']!=48):raise ValueError('Startup supervision v1 requires 8+48 observations')
    effective=c['world_size']*c['batch_sequences_per_gpu']*c['grad_accum_steps']
    if c['effective_sequences_per_step']!=effective:raise ValueError('Effective sequence count mismatch')
    if c['nominal_supervised_updates_per_step']!=effective*c['supervised_unroll_frames']:
        raise ValueError('Supervised target-frame count mismatch')
    if c.get('freeze_parent_steps',0)<0:raise ValueError('Invalid parent freeze duration')
    if c.get('augmentation',False):raise ValueError('v2 augmentation is not implemented; use explicit false')
    occlusion=c.get('temporal_occlusion_probability',0.)
    if not isinstance(occlusion,(int,float)) or not 0<=occlusion<=1:raise ValueError('Invalid temporal occlusion probability')
    if occlusion and c['supervised_unroll_frames']<14:raise ValueError('Temporal occlusion requires at least 14 supervised frames')
    startup=c.get('temporal_occlusion_startup_probability',0.)
    if not isinstance(startup,(int,float)) or not 0<=startup<=1:raise ValueError('Invalid startup occlusion probability')
    if startup and not occlusion:raise ValueError('Startup occlusion requires temporal occlusion enabled')
    real=c.get('real_initialization_probability',0.)
    if not isinstance(real,(int,float)) or not 0<=real<=1:raise ValueError('Invalid real initialization probability')
    prime=c.get('prime_initial_observation',False)
    if type(c.get('rotation_alignment',False)) is not bool:raise ValueError('Explicit boolean rotation_alignment required')
    if c.get('rotation_alignment',False) and c['architecture_id']!='stream_dual_cross_residual':raise ValueError('Rotation alignment needs residual architecture')
    if type(c.get('alignment_use_parent_latent',True)) is not bool:raise ValueError('Explicit boolean alignment latent switch required')
    spatial=c.get('alignment_spatial_weight',0.)
    if not isinstance(spatial,(int,float)) or not math.isfinite(spatial) or spatial<0:raise ValueError('Invalid spatial alignment weight')
    if (spatial or not c.get('alignment_use_parent_latent',True)) and not c.get('rotation_alignment',False):raise ValueError('Spatial/latent controls require rotation alignment')
    if spatial and not c.get('batch_current_features',False):raise ValueError('Spatial alignment v1 requires batched current features')
    if spatial and (not c.get('alignment_symmetric_object_ids') or not c.get('alignment_models_info_sha256')):raise ValueError('Spatial alignment requires bound symmetry metadata')
    if type(prime) is not bool or type(c.get('freeze_rgb_for_stage',False)) is not bool:raise ValueError('Explicit boolean initialization/freeze controls required')
    if prime and c['architecture_id']!='stream_dual_cross_residual':raise ValueError('Primed training currently supports the retained residual architecture')
    if real and (not prime or not c.get('external_initializers') or not c.get('external_initializers_sha256')):
        raise ValueError('Real initialization needs a primed, hash-bound train initializer manifest')
    return c


def config_hash(config):
    # Receipts/output locations do not define the learning trajectory.
    fields={k:v for k,v in config.items() if k not in ('preflight_receipt',)}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()


def make_model(c):
    if c.get('rotation_alignment',False):
        from lip.models.rotation_alignment import RotationAlignmentTracker
        return RotationAlignmentTracker(memory_frames=c['memory_frames'],dropout=c['dropout'],time_unit=c['time_unit'],max_gap_seconds=c['max_gap_seconds'],cache_kind=c.get('cache_kind','functional'),use_parent_latent=c.get('alignment_use_parent_latent',True),alignment_supervision=c.get('alignment_spatial_weight',0.)>0)
    if c['architecture_id']=='stream_rk_rotation_anchor_smooth':
        from lip.models.smooth_rotation_anchor import SmoothRotationAnchorRKTracker
        return SmoothRotationAnchorRKTracker(c['memory_frames'],c['dropout'],c['time_unit'],c['max_gap_seconds'],c['keyframe_slots'],c['keyframe_min_gap'],c['keyframe_max_age'],c['support_tolerance'],c['spatial_memory_side'],c['reference_write_limit'])
    if c['architecture_id']=='stream_rk_rotation_anchor':
        from lip.models.stream_rotation_anchor import RotationAnchorRKTracker
        return RotationAnchorRKTracker(c['memory_frames'],c['dropout'],c['time_unit'],c['max_gap_seconds'],c['keyframe_slots'],c['keyframe_min_gap'],c['keyframe_max_age'],c['support_tolerance'],c['spatial_memory_side'],c['reference_write_limit'])
    if c['architecture_id']=='stream_rk_adaptive_reference':
        from lip.models.stream_adaptive_reference import AdaptiveReferenceRKTracker
        return AdaptiveReferenceRKTracker(c['memory_frames'],c['dropout'],c['time_unit'],c['max_gap_seconds'],c['keyframe_slots'],c['keyframe_min_gap'],c['keyframe_max_age'],c['support_tolerance'],c['spatial_memory_side'],c['reference_write_limit'])
    if c['architecture_id']=='stream_rk_pose_reference':
        from lip.models.stream_pose_reference import PoseReferenceRKTracker
        return PoseReferenceRKTracker(c['memory_frames'],c['dropout'],c['time_unit'],c['max_gap_seconds'],c['keyframe_slots'],c['keyframe_min_gap'],c['keyframe_max_age'],c['support_tolerance'],c['spatial_memory_side'])
    if c['architecture_id']=='stream_rk_direct_pose':
        from lip.models.stream_direct_pose import DirectPoseRKTracker
        return DirectPoseRKTracker(c['memory_frames'],c['dropout'],c['time_unit'],c['max_gap_seconds'],c['keyframe_slots'],c['keyframe_min_gap'],c['keyframe_max_age'],c['support_tolerance'],c['spatial_memory_side'])
    if c['architecture_id']=='stream_rk_aligned':
        from lip.models.stream_aligned_memory import AlignedRKTracker
        return AlignedRKTracker(c['memory_frames'],c['dropout'],c['time_unit'],c['max_gap_seconds'],c['keyframe_slots'],c['keyframe_min_gap'],c['keyframe_max_age'],c['support_tolerance'],c['spatial_memory_side'],c['aligned_query_side'],c['aligned_sigma'])
    if c['architecture_id']=='stream_rk_spatial':
        from lip.models.stream_spatial_memory import SpatialRKTracker
        return SpatialRKTracker(c['memory_frames'],c['dropout'],c['time_unit'],c['max_gap_seconds'],c['keyframe_slots'],c['keyframe_min_gap'],c['keyframe_max_age'],c['support_tolerance'],c['spatial_memory_side'])
    if c['architecture_id']=='stream_rk_factorial':
        from lip.models.stream_rk import RKTracker
        return RKTracker(c['observation_reliability'],c['keyframe_memory'],c['memory_frames'],c['dropout'],
            c['time_unit'],c['max_gap_seconds'],c['keyframe_slots'],c['keyframe_min_gap'],c['keyframe_max_age'],c['support_tolerance'])
    from lip.models.stream_tracker import StreamTracker
    return StreamTracker(c['architecture_id'],c['memory_frames'],c['dropout'],False,c['time_unit'],c['max_gap_seconds'],c.get('cache_kind','functional'))


def optimizer_and_scheduler(model,c):
    groups={}
    for name,p in model.named_parameters():
        status=model.migration_status.get(name,{}).get('status','initialized')
        partial_new=model.migration_status.get(name,{}).get('partial_new',False)
        category='alignment' if name.startswith('rotation_alignment.') else 'quality' if name in ('reliability_strength','update_strength') else ('rgb' if name.startswith('rgb.') else ('loaded' if status in ('loaded','remapped') and not partial_new else 'new'))
        decay=p.ndim>1 and not name.endswith('bias')
        groups.setdefault((category,decay),[]).append((name,p))
    lrs=dict(alignment=c.get('lr_rotation_alignment',1e-4),rgb=c['lr_rgb_backbone'],loaded=c['lr_loaded_modules'],new=c['lr_new_modules'],quality=c.get('lr_quality',1e-3))
    opt=torch.optim.AdamW([dict(params=[p for n,p in values],names=[n for n,p in values],category=cat,
        lr=lrs[cat],weight_decay=c['weight_decay'] if decay else 0.) for (cat,decay),values in sorted(groups.items())])
    def factor(step,category):
        if category in ('loaded','rgb') and step<c.get('freeze_parent_steps',0):return 0.
        if category=='rgb' and step<c['freeze_rgb_steps']:return 0.
        warm=min(1.,(step+1)/max(1,c['warmup_steps']))
        ratio=min(1.,max(0.,(step-c['warmup_steps'])/max(1,c['max_stage_steps']-c['warmup_steps'])))
        return warm*(.1+.9*.5*(1+math.cos(math.pi*ratio)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,[lambda step,cat=g['category']:factor(step,cat) for g in opt.param_groups])
    return opt,scheduler


def verify_preflight(c,audit):
    path=Path(c['preflight_receipt'])
    if not path.exists():raise RuntimeError('This streaming architecture has no completed preflight: '+str(path))
    receipt=json.loads(path.read_text())
    from lip.engine.stream_checkpoint import source_hash
    expected=dict(architecture_id=c['architecture_id'],config_hash=config_hash(c),split_hash=audit['split_hash'],
                  mesh_hash=audit['mesh_hash'],cache_contract=cache_contract_for(c['architecture_id']),source_sha256=source_hash())
    for key,val in expected.items():
        if receipt.get(key)!=val:raise RuntimeError('Stale streaming preflight: '+key)
    if not receipt.get('approved'):raise RuntimeError('Streaming preflight has unresolved requirements')
    return receipt
