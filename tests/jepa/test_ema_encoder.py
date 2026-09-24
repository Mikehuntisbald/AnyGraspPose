import torch
from torch import nn
from lip.unified.ema_encoder import attach_ema, update_ema
from lip.engine.jepa_checkpoint import core_state, load_core
from lip.unified.optimized_training import make_optimizer


class Tiny(nn.Module):
    architecture_id = 'stream_cad_surface_jepa_v12'
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.core = nn.Linear(4, 4)


def config():
    return dict(ema_encoder=dict(momentum=.9),runtime=dict(reuse_teacher_real=False),
                training=dict(learning_rates=dict(encoder=1e-6,predictor=1e-5),weight_decay=.01))


def test_ema_gradient_optimizer_and_exact_formula():
    m=Tiny(); attach_ema(m,config()); opt=make_optimizer(m,config(),fused=False)
    before=m.ema_teacher.weight.detach().clone()
    x=torch.randn(2,3)
    loss=m.core(m.encoder(x)).square().mean();loss.backward()
    assert m.encoder.weight.grad.norm()>0
    assert all(p.grad is None for p in m.ema_teacher.parameters())
    assert any(g['category']=='encoder' for g in opt.param_groups)
    opt.step()
    assert torch.equal(m.ema_teacher.weight,before)
    online=m.encoder.weight.detach().clone();update_ema(m)
    assert torch.allclose(m.ema_teacher.weight,before*.9+online*.1)
    assert m.ema_updates.item()==1
    assert not torch.equal(m.encoder.weight,before)


def test_checkpoint_includes_both_encoders_and_strictly_restores():
    m=Tiny();attach_ema(m,config());update_ema(m)
    state={k:v.clone() for k,v in core_state(m).items()}
    assert 'encoder.weight' in state and 'ema_teacher.weight' in state and 'ema_updates' in state
    n=Tiny();attach_ema(n,config());load_core(n,state)
    assert all(torch.equal(v,core_state(n)[k]) for k,v in state.items())
    del state['encoder.weight']
    import pytest
    with pytest.raises(RuntimeError):load_core(n,state)


def test_online_target_reuse_is_rejected():
    import pytest
    c=config();c['runtime']['reuse_teacher_real']=True
    with pytest.raises(ValueError):attach_ema(Tiny(),c)


def test_teacher_changes_do_not_change_student_forward():
    m=Tiny();attach_ema(m,config());x=torch.randn(2,3)
    before=m.core(m.encoder(x)).detach()
    with torch.no_grad():m.ema_teacher.weight.add_(100)
    assert torch.equal(before,m.core(m.encoder(x)))


def test_position_resize_exact_forward_and_transpose_gradient():
    from lip.unified.ema_encoder import _PositionResize
    from torch.nn import functional as F
    x=torch.randn(1,4,9,9,requires_grad=True)
    options=dict(size=(4,4),antialias=True)
    basis=torch.eye(81).reshape(81,1,9,9)
    matrix=F.interpolate(basis,mode='bicubic',**options).flatten(1).T.contiguous()
    y=_PositionResize.apply(x,matrix,options)
    reference=F.interpolate(x,mode='bicubic',**options)
    assert torch.equal(y,reference)
    v=torch.randn_like(y)
    a=torch.autograd.grad(y,x,v)[0];b=torch.autograd.grad(reference,x,v)[0]
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
