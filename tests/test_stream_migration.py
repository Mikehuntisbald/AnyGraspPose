import copy
import random
import numpy as np
import pytest
import torch
from lip.models.tracker import Tracker
from lip.models.stream_tracker import StreamTracker
from lip.engine.stream_checkpoint import migrate,save,resume
from lip.engine.stream_config import optimizer_and_scheduler


def config():
    return dict(architecture_id='stream_single',memory_frames=8,lr_loaded_modules=1e-5,lr_new_modules=5e-5,
        lr_rgb_backbone=5e-6,weight_decay=.05,freeze_rgb_steps=2,warmup_steps=2,max_stage_steps=20)


def test_migration_is_complete_explicit_and_state_is_new(tmp_path):
    old=Tracker(False);path=tmp_path/'v1.pt'
    torch.save(dict(model=old.state_dict(),global_step=34700,config={},split_hash='s',mesh_hash='m'),path)
    new=StreamTracker();state_before=new.state[0].weight.detach().clone();report=migrate(new,path)
    assert report['coverage']>.98 and not report['optimizer_restored']
    assert report['parent']['global_step']==34700
    assert torch.equal(new.state[0].weight,state_before)
    assert not torch.equal(new.state[0].weight,old.state[0].weight)
    assert torch.equal(new.rgb[0].weight,old.rgb[0].weight)
    assert torch.equal(new.temporal.layers[2].self_attn.in_proj_weight,old.temporal.layers[2].self_attn.in_proj_weight)
    assert torch.equal(new.readout.query,old.readout)
    assert set(report['parameters'])==set(new.state_dict())
    single=tmp_path/'single.pt'
    torch.save(dict(model=new.state_dict(),architecture_id='stream_single',new_stage_step=3,config={},split_hash='s',mesh_hash='m'),single)
    dual=StreamTracker('stream_dual');r=migrate(dual,single)
    assert torch.equal(dual.readout.query[:,:1],new.readout.query)
    assert r['parameters']['readout.query']['loaded_numel']==256
    assert r['parameters']['readout.query']['initialized_numel']==256
    assert dual.readout.context_projection.weight.count_nonzero()==0
    with pytest.raises(ValueError):migrate(dual,path)


def test_strict_resume_optimizer_rng_sampler_and_trajectory(tmp_path):
    # Same checkpoint protocol on a small model: exercise actual Adam state, RNG,
    # group contract and a nontrivial uninterrupted-vs-resumed continuation.
    class Small(torch.nn.Module):
        def __init__(self):
            super().__init__();self.head=torch.nn.Linear(3,2);self.architecture_id='stream_single';self.migration_status={}
        def forward(self,x):return self.head(x)
    c=config();audit=dict(split_hash='s',mesh_hash='m');a=Small();opt,sched=optimizer_and_scheduler(a,c)
    def train(m,o,s):
        o.zero_grad();loss=m(torch.randn(4,3)).square().mean();loss.backward();o.step();s.step();return loss.detach()
    for _ in range(3):train(a,opt,sched)
    path=tmp_path/'last.pt';save(path,a,opt,sched,3,c,audit,192,None)
    expected_rng=torch.get_rng_state().clone();expected=[train(a,opt,sched) for _ in range(3)]
    b=Small();ob,sb=optimizer_and_scheduler(b,c);loaded=resume(path,b,ob,sb,audit,c)
    assert torch.equal(torch.get_rng_state(),expected_rng) and loaded['sampler_position']==192 and sb.last_epoch==3
    actual=[train(b,ob,sb) for _ in range(3)]
    assert all(torch.equal(x,y) for x,y in zip(actual,expected))
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    for change in [dict(memory_frames=16),dict(lr_new_modules=1e-3)]:
        with pytest.raises(ValueError):resume(path,b,ob,sb,audit,dict(c,**change))
    with pytest.raises(ValueError):resume(path,b,ob,sb,dict(audit,mesh_hash='wrong'),c)
