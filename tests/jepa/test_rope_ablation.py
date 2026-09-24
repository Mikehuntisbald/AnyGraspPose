import pytest
import torch
from lip.unified.rope_ablation import FrozenRoPESwitch,state_digest
from lip.unified.rope3d import enable_cad_rope3d


def test_switch_changes_only_gain_and_restores_on_exception():
    from test_cad_surface import model_sample
    model,obs=model_sample();enable_cad_rope3d(model)
    with torch.no_grad():model.cad_surface.rope3d.gain.fill_(.1)
    model.disable_history=True;model.requires_grad_(False).eval()
    switch=FrozenRoPESwitch(model);before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        with switch.arm(True):on,_=model(obs)
        with switch.arm(False):
            off,_=model(obs)
            assert not model.cad_surface.rope3d.gain.any()
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k!='cad_surface.rope3d.gain')
        assert not torch.equal(on['cad_match_log_prob'],off['cad_match_log_prob'])
        with pytest.raises(RuntimeError,match='probe interrupted'):
            with switch.arm(False):raise RuntimeError('probe interrupted')
    assert switch.verify(model)['model_state_unchanged']
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_switch_rejects_trainable_or_history_enabled_model():
    from test_cad_surface import model_sample
    model,_=model_sample();enable_cad_rope3d(model)
    with pytest.raises(ValueError,match='frozen'):FrozenRoPESwitch(model)
    model.requires_grad_(False).eval()
    with pytest.raises(ValueError,match='history'):FrozenRoPESwitch(model)


def test_compiled_attention_reads_updated_gate_tensor():
    from test_rope3d import fixture
    from lip.unified.rope3d import Gated3DRoPE
    reader,args,extras=fixture();reader.rope3d=Gated3DRoPE()
    reader.requires_grad_(False).eval()
    compiled=torch.compile(reader,backend='aot_eager',fullgraph=True)
    with torch.no_grad():
        reader.rope3d.gain.fill_(.1);on=compiled(*args,*extras)[1]['cad_match_log_prob'].clone()
        reader.rope3d.gain.zero_();off=compiled(*args,*extras)[1]['cad_match_log_prob'].clone()
        expected=reader(*args,*extras)[1]['cad_match_log_prob']
    assert torch.equal(off,expected) and not torch.equal(on,off)
