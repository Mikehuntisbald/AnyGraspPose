"""Explicit migration versus strict architecture-local optimizer/RNG resume."""
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from lip.engine.checkpoint import rng_state,restore_rng
from lip.engine.stream_state import cache_contract_for
from lip.engine.stream_config import config_hash


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    root=Path(__file__).resolve().parents[1];h=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def migrate(model,checkpoint,allow_untrained_single=False):
    old=torch.load(checkpoint,map_location='cpu',weights_only=False)
    src=old['model'];old_arch=old.get('architecture_id','legacy_v1')
    if old_arch=='legacy_v1' and model.architecture_id!='stream_single':raise ValueError('Migrate v1 to single first')
    if model.architecture_id=='stream_dual':
        if old_arch!='stream_single':raise ValueError('Dual warm-start requires stream_single')
        if old.get('new_stage_step',0)<1 and not allow_untrained_single:raise ValueError('Dual requires a trained single checkpoint')
    if model.architecture_id=='stream_dual_cross':
        if old_arch!='stream_dual' or old.get('new_stage_step',0)<1:raise ValueError('Cross readout requires a trained dual checkpoint')
    elif old_arch not in ('legacy_v1','stream_single'):raise ValueError('Use --resume within an existing architecture')
    target=model.state_dict();report={};used=set()
    for name,tensor in target.items():
        entry=dict(status='initialized',reason='new streaming parameter / buffer')
        source_name=name
        if old_arch=='legacy_v1' and name.startswith('readout.query'):source_name='readout'
        reset=(old_arch=='legacy_v1' and name.startswith(('state.','source_position.','temporal.time_bias.')))
        cross_reset=(model.architecture_id=='stream_dual_cross' and name.startswith('readout.') and name!='readout.query')
        if cross_reset:entry['reason']='new object-to-context attention; gate now consumes object and attention output'
        elif reset:entry['reason']='state semantics changed or new immutable-position/time-bias module'
        elif source_name in src and src[source_name].shape==tensor.shape:
            tensor.copy_(src[source_name]);used.add(source_name)
            entry=dict(status='remapped' if source_name!=name else 'loaded',source=source_name,reason='compatible module with unchanged parameter semantics')
        elif name=='readout.query' and model.architecture_id=='stream_dual' and 'readout.query' in src:
            tensor[:,:1].copy_(src['readout.query']);used.add('readout.query')
            entry=dict(status='remapped',source='readout.query',reason='object query copied; context query independently initialized',partial_new=True)
        copied=(tensor.numel()//2 if entry.get('partial_new') else tensor.numel()) if entry['status'] in ('loaded','remapped') else 0
        report[name]=dict(**entry,numel=tensor.numel(),loaded_numel=copied,initialized_numel=tensor.numel()-copied,shape=list(tensor.shape))
    # Strict load of a complete target dict; unmatched tensors were classified explicitly.
    model.load_state_dict(target,strict=True);model.migration_status=report
    skipped={name:dict(status='skipped',reason='old dual linear context fusion / gate replaced' if model.architecture_id=='stream_dual_cross' and name.startswith('readout.') else 'obsolete v1 state/time/readout representation or non-target module') for name in src if name not in used}
    param_names=set(dict(model.named_parameters()))
    loaded=sum(v['loaded_numel'] for k,v in report.items() if k in param_names)
    total=sum(p.numel() for p in model.parameters())
    required=('rgb.','rgb_proj.','geometry.','fusion.','head.','temporal.layers.','temporal.norm.')
    for prefix in required:
        if any(v['status']=='initialized' for n,v in report.items() if n.startswith(prefix)):
            raise ValueError('Required compatible module was not migrated: '+prefix)
    parent=dict(path=str(Path(checkpoint).resolve()),sha256=sha(checkpoint),architecture_id=old_arch,
        global_step=old.get('global_step'),new_stage_step=old.get('new_stage_step'),config=old.get('config'),
        split_hash=old['split_hash'],mesh_hash=old['mesh_hash'])
    result=dict(parent=parent,target_architecture=model.architecture_id,cache_contract=cache_contract_for(model.architecture_id),
                optimizer_restored=False,new_stage_step=0,parameters=report,source_skipped=skipped,
                loaded_parameter_numel=loaded,total_parameter_numel=total,coverage=loaded/total)
    return result


def save_init(path,model,config,migration):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    torch.save(dict(model=model.state_dict(),architecture_id=model.architecture_id,cache_contract=cache_contract_for(model.architecture_id),
        new_stage_step=0,global_step=0,config=config,config_hash=config_hash(config),parent=migration['parent'],
        split_hash=migration['parent']['split_hash'],mesh_hash=migration['parent']['mesh_hash'],
        migration_status=migration['parameters'],stage_kind='warm_start',source_sha256=source_hash()),path)
    (path.parent/'weight_migration.json').write_text(json.dumps(migration,indent=2))


def load_init(path,model,audit,config):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    for key,value in [('architecture_id',model.architecture_id),('cache_contract',cache_contract_for(model.architecture_id)),('split_hash',audit['split_hash']),('mesh_hash',audit['mesh_hash'])]:
        if obj.get(key)!=value:raise ValueError('Initialization mismatch: '+key)
    if obj.get('config',{}).get('memory_frames')!=config['memory_frames']:raise ValueError('Initialization memory capacity mismatch')
    model.load_state_dict(obj['model'],strict=True);model.migration_status=obj.get('migration_status',{})
    return obj


def group_contract(optimizer):
    return [{k:g[k] for k in ('names','category','weight_decay')} for g in optimizer.param_groups]


def save(path,model,optimizer,scheduler,step,config,audit,sampler_position,parent):
    states=[rng_state()]
    if dist.is_initialized():
        states=[None]*dist.get_world_size();dist.all_gather_object(states,rng_state())
        if dist.get_rank()!=0:return
    obj=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        architecture_id=model.architecture_id,cache_contract=cache_contract_for(model.architecture_id),new_stage_step=step,global_step=step,
        config=config,config_hash=config_hash(config),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],
        sampler_position=sampler_position,rng=states,parameter_groups=group_contract(optimizer),parent=parent,
        migration_status=model.migration_status,source_sha256=source_hash(),stage_kind='stream_training')
    temp=str(path)+'.tmp';torch.save(obj,temp);os.replace(temp,path)


def resume(path,model,optimizer,scheduler,audit,config,rank=0,world_size=1):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(architecture_id=model.architecture_id,cache_contract=cache_contract_for(model.architecture_id),
                  config_hash=config_hash(config),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'])
    for k,v in expected.items():
        if obj.get(k)!=v:raise ValueError('Resume mismatch: '+k)
    if obj.get('parameter_groups')!=group_contract(optimizer):raise ValueError('Resume optimizer parameter groups differ')
    if len(obj['rng'])!=world_size:raise ValueError('Resume world size changed')
    if obj['scheduler']['last_epoch']!=obj['new_stage_step']:raise ValueError('Scheduler stage step mismatch')
    model.load_state_dict(obj['model'],strict=True);optimizer.load_state_dict(obj['optimizer']);scheduler.load_state_dict(obj['scheduler'])
    restore_rng(obj['rng'][rank]);return obj
