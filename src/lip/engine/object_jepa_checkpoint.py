"""Architecture-local atomic checkpoints with all-rank RNG and sampler state."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from lip.engine.jepa_checkpoint import core_state,load_core,rng_state,restore_rng,sha,software_environment
from lip.jepa.config import config_hash


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def save(path,model,optimizer,scheduler,step,config,provenance):
    rank=dist.get_rank() if dist.is_initialized() else 0
    states=[rng_state()]
    if dist.is_initialized():
        states=[None]*dist.get_world_size();dist.all_gather_object(states,rng_state())
    if rank:return
    path=Path(path)
    record=dict(architecture_id=model.architecture_id,model_version='object-joint-v1',model=core_state(model),
        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,config=config,
        config_hash=config_hash(config),provenance=provenance,rng=states,
        sampler_position=step*config['training']['effective_batch'],software_environment=software_environment(),
        p0_only=not config['runtime']['formal'],state_carry_across_updates=False)
    torch.save(record,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)
    atomic_json(path.with_suffix('.receipt.json'),dict(completed=True,step=step,sha256=sha(path),
        sampler_position=record['sampler_position'],config_hash=record['config_hash'],all_rank_rng=len(states)))


def resume(path,model,optimizer,scheduler,config,provenance,rank,world):
    path=Path(path);receipt=json.loads(path.with_suffix('.receipt.json').read_text())
    if sha(path)!=receipt['sha256']:raise ValueError('Resume checkpoint hash mismatch')
    record=torch.load(path,map_location='cpu',weights_only=False)
    for key,expected in [('architecture_id',model.architecture_id),('config_hash',config_hash(config)),('provenance',provenance),('software_environment',software_environment())]:
        if record[key]!=expected:raise ValueError('Strict resume mismatch: '+key)
    if len(record['rng'])!=world or record['sampler_position']!=record['step']*config['training']['effective_batch']:
        raise ValueError('Resume world size/sampler mismatch')
    load_core(model,record['model']);optimizer.load_state_dict(record['optimizer']);scheduler.load_state_dict(record['scheduler'])
    restore_rng(record['rng'][rank]);return record
