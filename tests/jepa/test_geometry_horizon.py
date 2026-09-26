from copy import deepcopy
from pathlib import Path
import math
import pytest
import torch
import yaml
from lip.unified.geometry_horizon import validate_geometry_extension,extend_geometry
from lip.unified.horizon_resume import extension_factor,exact


def test_formal_geometry_extension_changes_only_output_and_horizon():
    root=Path(__file__).resolve().parents[2]
    old=yaml.safe_load((root/'configs/jepa/recovery_balance_v52_balanced.yaml').read_text())
    new=yaml.safe_load((root/'configs/jepa/recovery_formal_v53.yaml').read_text())
    validate_geometry_extension(old,new)
    for kind in ('loss','optimizer','batch'):
        bad=deepcopy(new)
        if kind=='loss':bad['cad_atlas']['correspondence_weight']=1.
        elif kind=='optimizer':bad['training']['learning_rates']['encoder']=1e-4
        else:bad['runtime']['microbatch']=8
        with pytest.raises(ValueError):validate_geometry_extension(old,bad)


def test_formal_geometry_schedule_preserves_boundary_without_lr_collapse():
    f=lambda step:extension_factor(step,200,100,5200,.5)
    assert f(200)==.5 and f(300)==1. and f(5200)==.5
    assert math.isclose(f(201),.505)
    assert .5<f(5199)<f(2700)<f(301)<1.


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Complete RNG contract includes CUDA')
def test_geometry_extension_preserves_adam_scheduler_and_rng(tmp_path):
    from lip.engine.jepa_checkpoint import rng_state,sha
    from lip.jepa.config import config_hash
    from lip.unified.checkpoint import software_environment
    class Tiny(torch.nn.Module):
        architecture_id='geometry-test'
        trainable_encoder=True
        def __init__(self):
            super().__init__();self.x=torch.nn.Parameter(torch.ones(2,device='cuda'))
    old=dict(paths={'output':'old'},geometry_transport_training={'updates':200},training={'effective_batch':32})
    provenance=dict(split_hash='split',mesh_hash='mesh',initializers_sha256='init',frozen_parameter_names=[])
    model=Tiny();opt=torch.optim.AdamW([dict(params=[model.x],names=['x'])],lr=1e-4)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:.5)
    for _ in range(200):
        opt.zero_grad();model.x.square().sum().backward();opt.step();sched.step()
    source=dict(config=old,config_hash=config_hash(old),architecture_id=model.architecture_id,software_environment=software_environment(),
        model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),step=200,sampler_position=6400,rng=[rng_state()],provenance=provenance)
    path=tmp_path/'source.pt';torch.save(source,path)
    new=deepcopy(old);new['paths']['output']='new';new['geometry_transport_training']['updates']=5200
    new['geometry_horizon_continuation']=dict(source_sha256=sha(path),source_config_hash=config_hash(old),source_step=200,rewarm_steps=100,floor=.5)
    restored=Tiny();ropt=torch.optim.AdamW([dict(params=[restored.x],names=['x'])],lr=1e-4)
    rsched=torch.optim.lr_scheduler.LambdaLR(ropt,lambda s:extension_factor(s,200,100,5200,.5))
    step,receipt=extend_geometry(path,restored,ropt,rsched,new,provenance,0,1)
    assert step==200 and receipt['verified'] and receipt['rng_exact']
    assert exact(ropt.state_dict(),source['optimizer']) and exact(rsched.state_dict(),source['scheduler'])
    ropt.zero_grad();restored.x.square().sum().backward();ropt.step();rsched.step()
    assert math.isclose(ropt.param_groups[0]['lr'],.505e-4)
    assert ropt.state[restored.x]['step'].item()==201
