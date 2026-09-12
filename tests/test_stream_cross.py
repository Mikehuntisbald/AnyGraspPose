import copy
from dataclasses import replace
import pytest
import torch
from test_stream_attention import metadata
from test_stream_geometry import fixture
from lip.models.stream_cross_readout import StreamCrossReadout
from lip.models.stream_tracker import StreamTracker
from lip.engine.stream_state import FrameMeta,CrossCache,TemporalCache,CROSS_CACHE_CONTRACT
from lip.engine.stream_checkpoint import migrate,load_init
from lip.geometry.renderer import Renderer
from lip.engine.stream_features import build_current_features,stack_current


def active_readout(window=8):
    m=StreamCrossReadout(window).eval()
    torch.nn.init.normal_(m.cross_attn.out_proj.weight,std=.04)
    return m


def test_cross_attention_and_gate_match_explicit_formula():
    torch.manual_seed(23);m=active_readout();z=torch.randn(1,4,2,256);meta=metadata(4,1);cache=()
    for i in range(4):out,cache=m(z[:,i],meta[i],cache)
    obj=z[:,-1,:1];ctx=z[:,:,1]
    h,_=m.cross_attn(obj,ctx,ctx,need_weights=False)
    gate=torch.sigmoid(m.gate(torch.cat((obj[:,0],h[:,0]),-1)))
    torch.testing.assert_close(out['cross_attention_output'],h[:,0],atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(out['latent'],obj[:,0]+gate*h[:,0],atol=2e-6,rtol=1e-5)
    assert not hasattr(m,'context_projection')
    original=StreamCrossReadout();initial,_=original(z[:,0],meta[0])
    assert torch.equal(initial['latent'],z[:,0,0])
    torch.testing.assert_close(initial['context_gate'],torch.full((1,1),torch.sigmoid(torch.tensor(-2.)).item()))


@pytest.mark.parametrize('frames',[1,2,8,9,32])
def test_cross_cached_full_reference_reset_padding_and_eviction(frames):
    torch.manual_seed(23);m=active_readout();z=torch.randn(2,frames,2,256);meta=metadata(frames,2,reset=True);cache=();outputs=[]
    with torch.no_grad():
        for i in range(frames):
            out,cache=m(z[:,i],meta[i],cache);outputs.append(out['latent'])
            assert len(cache)<=8 and cache[-1].key.shape==(2,8,1,32)
        ref,_,_=m.full_reference(z,meta)
    error=float((torch.stack(outputs,1)-ref).abs().max());print('cross_reference',frames,error)
    torch.testing.assert_close(torch.stack(outputs,1),ref,atol=2e-6,rtol=1e-5)


def test_cross_gradients_reach_past_context_and_match_reference():
    torch.manual_seed(15);a=active_readout(3);b=copy.deepcopy(a)
    x=torch.randn(1,7,2,256,requires_grad=True);y=x.detach().clone().requires_grad_(True);meta=metadata(7,1);cache=()
    for i in range(7):out,cache=a(x[:,i],meta[i],cache)
    out['latent'].square().mean().backward();ref,_,_=b.full_reference(y,meta);ref[:,-1].square().mean().backward()
    assert x.grad[:,4,1].abs().sum()>0 and x.grad[:,:4].abs().sum()==0 and x.grad[:,:-1,0].abs().sum()==0
    torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=1e-4)
    largest=0.
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        if name=='query':assert p.grad is None and q.grad is None;continue
        assert name==other and p.grad is not None and q.grad is not None
        largest=max(largest,float((p.grad-q.grad).abs().max()))
        torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-4)
    print('cross_gradient_max_abs',largest)


def test_context_is_query_conditioned_and_causal_and_object_never_cached():
    torch.manual_seed(9);m=active_readout();z=torch.randn(1,12,2,256);meta=metadata(12,1)
    with torch.no_grad():
        a,_,_=m.full_reference(z,meta);changed=z.clone();changed[:,7:]*=100
        b,_,_=m.full_reference(changed,meta)
        torch.testing.assert_close(a[:,:7],b[:,:7],atol=0,rtol=0)
        cache=();altered=()
        for i in range(12):
            out,cache=m(z[:,i],meta[i],cache)
            v=z[:,i].clone();v[:,0]+=torch.randn_like(v[:,0])*10
            different,altered=m(v,meta[i],altered)
            assert torch.equal(cache[-1].key,altered[-1].key) and torch.equal(cache[-1].value,altered[-1].value)
        assert not torch.allclose(out['cross_attention_output'],different['cross_attention_output'])
        future=replace(meta[0],timestamp=meta[-1].timestamp+10)
        invalid_old=m.project_context(torch.ones(1,1,256)*100,future)
        fresh,_=m(z[:,-1],meta[-1]);guarded,_=m(z[:,-1],meta[-1],(invalid_old,))
        torch.testing.assert_close(fresh['latent'],guarded['latent'])


def test_cross_invalid_context_and_reset_lifecycle():
    mesh,pose,k=fixture();model=StreamTracker('stream_dual_cross').eval();renderer=Renderer('cpu')
    state=model.initialize(pose,mesh,k,'stream',0.,image_shape=(64,64))
    assert isinstance(state.cache,CrossCache) and state.cache_contract==CROSS_CACHE_CONTRACT
    rgb=torch.rand(3,64,64);depth=torch.full((1,64,64),.6)
    proposal,pending=model.step(rgb,depth,.03,state,renderer=renderer,image_size=32)
    assert proposal['status']=='ok' and not state.cache.contexts and len(pending.cache.contexts)==1
    state=model.commit(proposal,pending);detached=state.detach()
    assert isinstance(detached.cache,CrossCache) and detached.cache.contexts[0].key.grad_fn is None
    assert torch.equal(detached.cache.contexts[0].key,state.cache.contexts[0].key)
    bad,unchanged=model.step(rgb*float('nan'),depth,.06,state,renderer=renderer,image_size=32)
    assert bad['status']=='invalid_input' and unchanged is state
    mismatch=replace(state,cache_contract='wrong')
    bad,_=model.step(rgb,depth,.06,mismatch,renderer=renderer,image_size=32)
    assert bad['status']=='cache_contract_changed'
    reset=model.correct(state,pose,relocalization=True)
    assert reset.generation==state.generation+1 and not reset.cache.contexts
    original=model.readout.project_context
    def broken(*args):
        block=original(*args);return replace(block,value=block.value*float('nan'))
    model.readout.project_context=broken
    bad,unchanged=model.step(rgb,depth,.06,state,renderer=renderer,image_size=32)
    assert bad['status']=='nonfinite_proposal' and unchanged is state


def test_dual_migration_resets_gate_semantics_and_preserves_compatible_weights(tmp_path):
    old=StreamTracker('stream_dual');old.readout.gate[-1].bias.data.fill_(3.)
    path=tmp_path/'dual.pt';torch.save(dict(model=old.state_dict(),architecture_id='stream_dual',new_stage_step=8800,split_hash='s',mesh_hash='m',config={'memory_frames':8}),path)
    new=StreamTracker('stream_dual_cross');report=migrate(new,path)
    assert report['coverage']>.95 and report['cache_contract']==CROSS_CACHE_CONTRACT and not report['optimizer_restored']
    assert torch.equal(new.readout.query,old.readout.query) and torch.equal(new.head[-1].weight,old.head[-1].weight)
    assert new.readout.gate[-1].bias.item()==-2 and new.readout.cross_attn.out_proj.weight.count_nonzero()==0
    assert report['parameters']['readout.gate.0.weight']['status']=='initialized'
    assert 'readout.context_projection.weight' in report['source_skipped']
    with pytest.raises(ValueError):load_init(path,new,dict(split_hash='s',mesh_hash='m'),dict(memory_frames=8))


def test_new_cross_residual_does_not_pollute_source_or_context_cache():
    mesh,pose,k=fixture();model=StreamTracker('stream_dual_cross').eval()
    f,d=build_current_features(torch.rand(3,64,64),torch.full((1,64,64),.6),pose,k,mesh,Renderer('cpu'),.03,0.,size=32)
    meta=metadata(1,1)[0]
    with torch.no_grad():
        a,ca=model(stack_current([f]),meta)
        model.readout.cross_attn.out_proj.weight.normal_(std=.1)
        b,cb=model(stack_current([f]),meta)
    assert not torch.equal(a['latent'],b['latent'])
    for la,lb in zip(ca.layers,cb.layers):
        assert torch.equal(la[-1].key,lb[-1].key) and torch.equal(la[-1].value,lb[-1].value)
    assert torch.equal(ca.contexts[-1].key,cb.contexts[-1].key)


def test_zero_initialized_residual_learns_qkv_and_gate():
    torch.manual_seed(82);m=StreamCrossReadout();optimizer=torch.optim.AdamW(m.parameters(),lr=.01)
    z=torch.randn(1,4,2,256);meta=metadata(4,1)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True);cache=()
        for i in range(4):out,cache=m(z[:,i],meta[i],cache)
        out['latent'].square().mean().backward()
        optimizer.step()
    assert m.cross_attn.in_proj_weight.grad[:256].abs().sum()>0
    assert m.cross_attn.in_proj_weight.grad[256:512].abs().sum()>0
    assert m.gate[0].weight.grad.abs().sum()>0
    assert out['cross_attention_output'].abs().sum()>0
