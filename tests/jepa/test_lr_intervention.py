from copy import deepcopy
import torch
from lip.unified.lr_intervention import rate_factor,apply_rates
from lip.unified.horizon_resume import extension_factor,exact


def config(start=25400,scale=.3):
    return dict(migration=dict(source_step=start),lr_intervention=dict(scale=scale,schedule_origin=25400,rewarm_steps=200,schedule_end=35400,floor=.1))


def test_global_clock_survives_new_stage():
    for step in (26500,26503,30400,35400):
        assert rate_factor(config(26500),step-26500)==.3*extension_factor(step,25400,200,35400,.1)


def test_only_lr_changes_adam_state():
    p=torch.nn.Parameter(torch.tensor([2.,-1.]));opt=torch.optim.AdamW([p],lr=1e-4)
    p.square().sum().backward();opt.step()
    source={'optimizer':deepcopy(opt.state_dict())}
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda x:.3)
    opt.load_state_dict(source['optimizer'])
    receipt=apply_rates(opt,sched,source,config())
    assert receipt['new']==[3e-5] and receipt['non_lr_adam_exact']
    assert exact(opt.state_dict()['state'],source['optimizer']['state'])
    assert opt.param_groups[0]['lr']==3e-5


def test_resume_keeps_scaled_schedule():
    p=torch.nn.Parameter(torch.ones(1));o=torch.optim.AdamW([p],lr=1e-4)
    s=torch.optim.lr_scheduler.LambdaLR(o,lambda x:rate_factor(config(),x))
    for _ in range(3):o.step();s.step()
    state=deepcopy(o.state_dict());clock=deepcopy(s.state_dict())
    q=torch.nn.Parameter(torch.ones(1));oo=torch.optim.AdamW([q],lr=1e-4)
    ss=torch.optim.lr_scheduler.LambdaLR(oo,lambda x:rate_factor(config(),x))
    oo.load_state_dict(state);ss.load_state_dict(clock)
    o.step();s.step();oo.step();ss.step()
    assert s.get_last_lr()==ss.get_last_lr()
