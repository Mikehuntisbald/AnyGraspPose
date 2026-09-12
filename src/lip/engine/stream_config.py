import hashlib
import json
import math
from pathlib import Path
import torch
import yaml
from lip.engine.stream_state import cache_contract_for


def load_stream_config(path):
    c=yaml.safe_load(Path(path).read_text())
    if c['architecture_id'] not in ('stream_single','stream_dual','stream_dual_cross'):raise ValueError('Unknown architecture')
    if c['architecture_id']=='stream_dual_cross' and c.get('cache_kind','functional')!='functional':raise ValueError('Cross cache requires functional mode')
    fixed=dict(source_tokens=17,latent_dim=256,temporal_layers=4,attention_heads=8,image_size=224)
    for k,v in fixed.items():
        if c.get(k)!=v:raise ValueError(f'v2 contract requires {k}={v}')
    if c['memory_frames'] not in (1,8,16):raise ValueError('Use a separately trained W=1,8,16 config')
    if c['precision'] not in ('fp32','bf16'):raise ValueError('Explicit precision required')
    if c.get('basin_weight',0) or c.get('foundationpose_enabled',False) or c.get('depth_correction',False):
        raise ValueError('FP, basin loss and GT-depth correction are not v2 defaults')
    if c['burn_in_frames']<0 or c['supervised_unroll_frames']<1:raise ValueError('Invalid unroll')
    effective=c['world_size']*c['batch_sequences_per_gpu']*c['grad_accum_steps']
    if c['effective_sequences_per_step']!=effective:raise ValueError('Effective sequence count mismatch')
    if c['nominal_supervised_updates_per_step']!=effective*c['supervised_unroll_frames']:
        raise ValueError('Supervised target-frame count mismatch')
    if c.get('augmentation',False):raise ValueError('v2 augmentation is not implemented; use explicit false')
    return c


def config_hash(config):
    # Receipts/output locations do not define the learning trajectory.
    fields={k:v for k,v in config.items() if k not in ('preflight_receipt',)}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()


def make_model(c):
    from lip.models.stream_tracker import StreamTracker
    return StreamTracker(c['architecture_id'],c['memory_frames'],c['dropout'],False,c['time_unit'],c['max_gap_seconds'],c.get('cache_kind','functional'))


def optimizer_and_scheduler(model,c):
    groups={}
    for name,p in model.named_parameters():
        status=model.migration_status.get(name,{}).get('status','initialized')
        partial_new=model.migration_status.get(name,{}).get('partial_new',False)
        category='rgb' if name.startswith('rgb.') else ('loaded' if status in ('loaded','remapped') and not partial_new else 'new')
        decay=p.ndim>1 and not name.endswith('bias')
        groups.setdefault((category,decay),[]).append((name,p))
    lrs=dict(rgb=c['lr_rgb_backbone'],loaded=c['lr_loaded_modules'],new=c['lr_new_modules'])
    opt=torch.optim.AdamW([dict(params=[p for n,p in values],names=[n for n,p in values],category=cat,
        lr=lrs[cat],weight_decay=c['weight_decay'] if decay else 0.) for (cat,decay),values in sorted(groups.items())])
    def factor(step,category):
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
