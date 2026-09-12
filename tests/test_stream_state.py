import torch
import pytest
from test_stream_geometry import fixture
from lip.models.stream_tracker import StreamTracker
from lip.geometry.renderer import Renderer


def test_transaction_lifecycle_one_image_and_no_cache_contamination():
    mesh,t,k=fixture();m=StreamTracker(dropout=0.).eval();state=m.initialize(t,mesh,k,'stream',10.,image_shape=(64,64))
    calls=[];handles=[m.rgb.register_forward_pre_hook(lambda _,a:calls.append(('rgb',a[0].shape[0]))),
                     m.geometry.register_forward_pre_hook(lambda _,a:calls.append(('geo',a[0].shape[0])))]
    rgb=torch.zeros(3,64,64);depth=torch.zeros(1,64,64);renderer=Renderer('cpu')
    p,n=m.step(rgb,depth,10.03,state,renderer=renderer,image_size=32)
    assert p['status']=='ok' and calls==[('rgb',1),('geo',1)]
    assert len(state.cache.metadata)==0 and len(n.cache.metadata)==1
    assert torch.equal(state.pose_centered,n.pose_centered)
    state=m.commit(p,n);old=state.source_metadata[0].A_source.clone();kv=state.cache.layers[0][0].key.clone()
    small=p['pose_original'].clone();small[0,3]+=.005;corrected=m.correct(state,small)
    assert torch.equal(corrected.cache.layers[0][0].key,kv)
    p,n=m.step(rgb,depth,10.06,corrected,renderer=renderer,image_size=32)
    assert torch.equal(n.source_metadata[0].A_source,old)
    assert not torch.equal(n.source_metadata[1].T_base_source,state.source_metadata[0].T_base_source)
    for kwargs in [dict(stream_id='new'),dict(camera_id='new'),dict(object_id='new'),dict(mesh_hash='new'),dict(K=k+1)]:
        fail,unchanged=m.step(rgb,depth,10.06,state,**kwargs)
        assert fail['needs_reinit'] and unchanged is state
    for stamp in [9.,10.03,11.]:
        fail,unchanged=m.step(rgb,depth,stamp,state);assert fail['needs_reinit'] and unchanged is state
    badrgb=rgb.clone();badrgb[0,0,0]=float('nan')
    fail,unchanged=m.step(badrgb,depth,10.06,state,renderer=renderer,image_size=32)
    assert fail['status']=='invalid_input' and unchanged is state
    m.load_state_dict(m.state_dict());fail,_=m.step(rgb,depth,10.06,state);assert fail['status']=='weights_changed'
    reset=m.correct(state,small,relocalization=True);assert not reset.cache.metadata and reset.generation==state.generation+1
    for h in handles:h.remove()


def test_nonfinite_proposal_does_not_commit_cache():
    mesh,t,k=fixture();m=StreamTracker().eval();m.head[-1].bias.data.fill_(float('nan'))
    state=m.initialize(t,mesh,k,'s',0,image_shape=(64,64))
    p,n=m.step(torch.zeros(3,64,64),torch.zeros(1,64,64),.03,state,renderer=Renderer('cpu'),image_size=32)
    assert p['status']=='nonfinite_proposal' and n is state and not state.cache.metadata
    with pytest.raises(ValueError):m.commit(p,n)
    with pytest.raises(ValueError):m.initialize(None,mesh,k,'s',0)
