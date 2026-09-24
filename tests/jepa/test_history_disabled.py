import torch
from test_cad_surface import model_sample


def test_no_history_ignores_incoming_memory_skips_writer_and_gets_no_gradients():
    model,obs=model_sample()
    with torch.no_grad():_,memory=model(obs)
    model.disable_history=True
    for block in model.core.blocks:block.disable_history=True
    def forbidden(*args):raise AssertionError('Disabled history was executed')
    hooks=[model.writer.register_forward_pre_hook(forbidden)]
    hooks += [b.history.register_forward_pre_hook(forbidden) for b in model.core.blocks]
    a,ma=model(obs,memory,torch.ones(1,dtype=torch.bool))
    b,mb=model(obs,None,torch.zeros(1,dtype=torch.bool))
    assert torch.equal(a['patch_latent'],b['patch_latent'])
    assert not ma.objects and not ma.contexts and not mb.objects and not mb.contexts
    assert a['read_memory_tokens']==0
    a['f_predicted'].square().mean().backward()
    assert all(p.grad is None for p in model.writer.parameters())
    assert all(p.grad is None for block in model.core.blocks for p in block.history.parameters())
    for h in hooks:h.remove()
