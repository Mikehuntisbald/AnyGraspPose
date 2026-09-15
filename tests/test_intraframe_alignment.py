from dataclasses import replace
import pytest
import torch
from test_stream_geometry import fixture
from lip.engine.intraframe import step_iterated
from lip.engine.stream_features import build_current_features,stack_current
from lip.engine.stream_state import FrameMeta
from lip.geometry.renderer import Renderer
from lip.models.stream_tracker import StreamTracker
from lip.models.rotation_alignment import RotationAlignmentTracker


def test_two_pass_commits_one_history_and_keeps_actual_motion():
    torch.set_num_threads(2);mesh,t,k=fixture();model=StreamTracker('stream_dual_cross_residual').eval()
    model.head[-1].bias.data.copy_(torch.tensor([.01,.02,.03,.01,0,0]))
    state=model.initialize(t,mesh,k,'s',0.,image_shape=(64,64));renderer=Renderer('cpu')
    rgb=torch.rand(3,64,64);dep=torch.ones(1,64,64)*.6
    first,pending=model.step(rgb,dep,.03,state,renderer=renderer,image_size=32)
    one,onepending=step_iterated(model,rgb,dep,.03,state,renderer=renderer,image_size=32)
    assert torch.equal(first['pose_centered'],one['pose_centered'])
    p,n=step_iterated(model,rgb,dep,.03,state,iterations=2,renderer=renderer,image_size=32)
    assert p['inner_iterations_accepted']==2 and state.frame_id==0 and not state.cache.metadata
    assert n.frame_id==1 and len(n.cache.metadata)==len(n.cache.contexts)==len(n.source_metadata)==1
    assert torch.equal(n.source_metadata[0].T_base_source,first['pose_centered'])
    committed=model.commit(p,n);assert torch.equal(committed.previous_pose,state.pose_centered)
    assert committed.previous_timestamp==0. and committed.timestamp==.03
    f,_=build_current_features(rgb,dep,committed.pose_centered,k,committed.mesh,renderer,.06,.03,
        state.pose_centered,0.,size=32,motion_base=committed.pose_centered)
    altered=committed.pose_centered.clone();altered[0,3]+=.1
    other,_=build_current_features(rgb,dep,altered,k,committed.mesh,renderer,.06,.03,
        state.pose_centered,0.,size=32,motion_base=committed.pose_centered)
    assert torch.equal(f['state_input'][14:23],other['state_input'][14:23])
    p,n=step_iterated(model,rgb,dep,.06,committed,iterations=2,renderer=renderer,image_size=32)
    assert n.frame_id==2 and len(n.cache.metadata)==2 and len(committed.cache.metadata)==1


def test_failed_second_pass_falls_back_to_first_without_duplicate_history(monkeypatch):
    mesh,t,k=fixture();model=StreamTracker('stream_dual_cross_residual').eval();s=model.initialize(t,mesh,k,'s',0.,image_shape=(64,64))
    original=model.step
    def injected(*args,**kw):
        if kw.get('refinement_base') is not None:return dict(status='injected_failure'),s
        return original(*args,**kw)
    monkeypatch.setattr(model,'step',injected)
    p,n=step_iterated(model,torch.zeros(3,64,64),torch.ones(1,64,64)*.6,.03,s,iterations=2,image_size=32)
    assert p['status']=='ok' and p['inner_failure']=='injected_failure' and p['inner_iterations_accepted']==1
    assert len(model.commit(p,n).cache.metadata)==1
    with pytest.raises(ValueError):step_iterated(model,None,None,0,s,iterations=3)


def test_alignment_zero_start_and_nonzero_rotation_only_gradient():
    torch.set_num_threads(2);torch.manual_seed(4);mesh,t,k=fixture();parent=StreamTracker('stream_dual_cross_residual').eval()
    model=RotationAlignmentTracker().eval();target=model.state_dict();target.update(parent.state_dict());model.load_state_dict(target)
    features,_=build_current_features(torch.rand(3,64,64),torch.ones(1,64,64)*.6,t,k,mesh,Renderer('cpu'),.03,0.,size=224)
    meta=FrameMeta(torch.tensor([.03],dtype=torch.float64),torch.tensor([1]),torch.tensor([0]),torch.ones(1,17,dtype=torch.bool),torch.zeros(1,17))
    f=stack_current([features]);old,_=parent(f,meta);new,_=model(f,meta)
    assert torch.equal(old['pose_centered'],new['pose_centered']) and torch.equal(old['latent'],new['latent'])
    loss=new['pose_centered'][0,0,1];loss.backward()
    assert model.rotation_alignment.output[-1].weight.grad.norm()>0
    with torch.no_grad():model.rotation_alignment.output[-1].bias.copy_(torch.tensor([.1,0.,0.]))
    out,_=model(f,meta)
    assert torch.equal(out['pose_centered'][:,:3,3],old['pose_centered'][:,:3,3])
    assert not torch.equal(out['pose_centered'][:,:3,:3],old['pose_centered'][:,:3,:3])
