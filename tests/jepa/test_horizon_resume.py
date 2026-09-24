from copy import deepcopy
from pathlib import Path
import math
import pytest
import yaml
import torch
from lip.unified.horizon_resume import extension_factor,validate_extension_config,exact


def test_extension_starts_at_saved_lr_and_rewarms_then_decays():
    f=lambda s:extension_factor(s,500,100,5000,.1)
    assert f(500)==.1 and f(600)==1. and math.isclose(f(5000),.1)
    assert .1<f(501)<f(599)<1
    assert 1>f(601)>f(2500)>f(4999)>.1


def test_optimizer_and_scheduler_state_survive_lambda_change():
    x=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.AdamW([x],lr=1e-4)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda epoch:.1)
    x.square().sum().backward();opt.step();sched.step()
    state=deepcopy(opt.state_dict());clock=deepcopy(sched.state_dict())
    clock.update(last_epoch=500,_step_count=501)
    new=torch.optim.AdamW([torch.nn.Parameter(x.detach().clone())],lr=1e-4)
    ns=torch.optim.lr_scheduler.LambdaLR(new,lambda s:extension_factor(s,500,100,5000,.1))
    new.load_state_dict(state);ns.load_state_dict(clock)
    assert exact(new.state_dict(),state) and exact(ns.state_dict(),clock)
    new.step();ns.step()
    assert math.isclose(new.param_groups[0]['lr'],1.09e-5)


def test_extension_rejects_unrelated_training_changes():
    root=Path(__file__).resolve().parents[2]
    old=yaml.safe_load((root/'configs/jepa/two_stream_v9_clean.yaml').read_text())
    new=yaml.safe_load((root/'configs/jepa/two_stream_v9_to5000.yaml').read_text())
    validate_extension_config(old,new)
    changed=deepcopy(new);changed['training']['loss_weights']['real_feature']=9
    with pytest.raises(ValueError):validate_extension_config(old,changed)
    changed=deepcopy(new);changed['runtime']['microbatch']=8
    with pytest.raises(ValueError):validate_extension_config(old,changed)


def test_phase_relative_clock_extension_preserves_boundary_and_next_lr():
    from lip.unified.horizon_resume import scheduler_origin
    config={'runtime':{'phase_relative_schedule':True},'migration':{'source_step':650}}
    origin=scheduler_origin(config)
    assert origin==650
    assert scheduler_origin({'runtime':{},'migration':{'source_step':500}})==0
    x=torch.nn.Parameter(torch.tensor([1.]))
    opt=torch.optim.AdamW([x],lr=3e-5)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:.1)
    opt.step();sched.step()
    state=deepcopy(opt.state_dict());clock=deepcopy(sched.state_dict())
    clock.update(last_epoch=128,_step_count=129)
    new=torch.optim.AdamW([torch.nn.Parameter(x.detach().clone())],lr=3e-5)
    ns=torch.optim.lr_scheduler.LambdaLR(new,lambda e:extension_factor(e+origin,778,100,5000,.1))
    new.load_state_dict(state);ns.load_state_dict(clock)
    assert exact(new.state_dict(),state) and exact(ns.state_dict(),clock)
    assert ns.last_epoch+origin==778
    assert math.isclose(new.param_groups[0]['lr'],3e-6)
    new.step();ns.step()
    assert math.isclose(new.param_groups[0]['lr'],3e-5*.109)


def test_recovery_25000_extension_preserves_objective_and_uses_original_clock():
    root=Path(__file__).resolve().parents[2]
    old=yaml.safe_load((root/'configs/jepa/local_difference_v15.yaml').read_text())
    new=yaml.safe_load((root/'configs/jepa/local_difference_v15_to25000.yaml').read_text())
    validate_extension_config(old,new)
    from lip.unified.horizon_resume import scheduler_origin
    assert scheduler_origin(new)==11000
    f=lambda epoch:extension_factor(epoch+scheduler_origin(new),12000,100,25000,.1)
    assert f(1000)==.1 and f(1100)==1 and f(14000)==.1
    assert f(1001)>.1 and f(4000)>.8
    # The recovery branch must not shadow the horizon schedule or restore path.
    import ast
    tree=ast.parse((root/'tools/train_two_stream.py').read_text())
    rate=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='rate')
    assert 'horizon_continuation' in ast.unparse(rate.body[0].test)
