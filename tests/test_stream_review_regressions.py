import importlib.util
from pathlib import Path
from dataclasses import replace
import pytest
import torch
from test_stream_geometry import fixture
from lip.models.stream_tracker import StreamTracker
from lip.engine import stream_runtime
from lip.engine.stream_state import FrameMeta
from lip.evaluate_stream import recovery_count


def test_default_renderer_reused_across_frames_and_explicit_override(monkeypatch):
    mesh,t,k=fixture();model=StreamTracker().eval()
    real=stream_runtime.Renderer;created=[]
    def factory(device):
        renderer=real(device);created.append(renderer);return renderer
    monkeypatch.setattr(stream_runtime,'Renderer',factory)
    state=model.initialize(t,mesh,k,'s',0.,image_shape=(64,64))
    rgb=torch.zeros(3,64,64);depth=torch.zeros(1,64,64)
    for i in range(1,4):
        p,n=model.step(rgb,depth,i*.03,state,image_size=32)
        assert p['status']=='ok';state=model.commit(p,n)
    assert len(created)==1
    p,n=model.step(rgb,depth,.12,state,image_size=32,renderer=real('cpu'))
    assert p['status']=='ok' and len(created)==1


@pytest.mark.parametrize('bad', ['reflection','nonrigid','bottom','behind','nan'])
def test_invalid_pose_initialize_and_commit(bad):
    mesh,t,k=fixture();model=StreamTracker().eval();state=model.initialize(t,mesh,k,'s',0.)
    invalid=t.clone()
    if bad=='reflection':invalid[0,0]=-1
    elif bad=='nonrigid':invalid[0,0]=2
    elif bad=='bottom':invalid[3,0]=.1
    elif bad=='behind':invalid[2,3]=-.1
    else:invalid[0,0]=float('nan')
    with pytest.raises(ValueError):model.initialize(invalid,mesh,k,'s',0.)
    with pytest.raises(ValueError):model.correct(state,invalid)
    pending=replace(state,pending_pose=invalid,pending_timestamp=.03)
    with pytest.raises(ValueError):model.commit(dict(status='ok',pose_centered=invalid),pending)


def test_finite_behind_camera_prediction_does_not_advance_state():
    mesh,t,k=fixture();model=StreamTracker().eval();model.head[-1].bias.data[5]=-100
    state=model.initialize(t,mesh,k,'s',0.,image_shape=(64,64))
    p,n=model.step(torch.zeros(3,64,64),torch.zeros(1,64,64),.03,state,image_size=32)
    assert p['status']=='invalid_proposal_pose' and n is state and not state.cache.metadata


@pytest.mark.parametrize('architecture',['stream_single','stream_dual','stream_dual_cross','stream_dual_cross_residual'])
def test_benchmark_reference_matches_cached_readout(architecture):
    path=Path(__file__).resolve().parents[1]/'tools/benchmark_stream.py'
    spec=importlib.util.spec_from_file_location('benchmark_stream_review',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(3);model=StreamTracker(architecture,memory_frames=2).eval()
    if architecture in ('stream_dual_cross','stream_dual_cross_residual'):model.readout.cross_attn.out_proj.weight.data.normal_(std=.03)
    sources=[];metadata=[];cache=None;contexts=()
    with torch.no_grad():
        for i in range(4):
            source=torch.randn(1,17,256);sources.append(source)
            meta=FrameMeta(torch.tensor([i*.03],dtype=torch.float64),torch.tensor([i]),torch.tensor([0]),torch.ones(1,17,dtype=torch.bool),torch.zeros(1,17));metadata.append(meta)
            query=model.readout.query+model.token_type[2]
            z,cache=model.temporal(source,query,meta,cache,model.readout.dual)
            if architecture in ('stream_dual_cross','stream_dual_cross_residual'):out,contexts=model.readout(z,meta,contexts)
            else:out=model.readout(z)
            reference=module.reference_latent(model,sources,metadata)
            torch.testing.assert_close(reference,out['latent'],atol=2e-5,rtol=2e-5)


def test_recovery_requires_success_and_does_not_cross_initialization_or_stream():
    def row(i,stream='a',lost=False,success=False,status='ok',init=False,needs=False):
        return dict(frame_index=i,stream_id=stream,lost=lost,adds_01=success,status=status,initialization=init,needs_reinit=needs)
    assert recovery_count([row(0,lost=True),row(1),row(2,success=True)])==1
    assert recovery_count([row(0,lost=True),row(1,success=True,status='invalid_input')])==0
    assert recovery_count([row(0,lost=True),row(1,success=True,needs=True)])==0
    assert recovery_count([row(0,lost=True),row(1,init=True,success=True),row(2,success=True)])==0
    assert recovery_count([row(0,lost=True),row(1,stream='b',success=True)])==0
    assert recovery_count([row(2,success=True),row(0,lost=True),row(1)])==1


def test_default_renderer_device_change_replaces_resource(monkeypatch):
    from types import SimpleNamespace
    created=[]
    def factory(device):
        obj=object();created.append((device,obj));return obj
    monkeypatch.setattr(stream_runtime,'Renderer',factory)
    model=SimpleNamespace()
    first=stream_runtime.default_renderer(model,'cpu')
    assert stream_runtime.default_renderer(model,torch.device('cpu')) is first
    second=stream_runtime.default_renderer(model,'cuda:0')
    assert second is not first and len(created)==2
    assert stream_runtime.default_renderer(model,'cuda:0') is second


def test_invalid_state_rejected_before_renderer_creation(monkeypatch):
    mesh,t,k=fixture();model=StreamTracker().eval()
    state=model.initialize(t,mesh,k,'s',0.,image_shape=(64,64))
    invalid=state.pose_centered.clone();invalid[0,0]=-1
    state=replace(state,pose_centered=invalid)
    def forbidden(device):raise AssertionError('Must validate before allocating renderer')
    monkeypatch.setattr(stream_runtime,'Renderer',forbidden)
    p,n=model.step(torch.zeros(3,64,64),torch.zeros(1,64,64),.03,state)
    assert p['status']=='invalid_state_pose' and n is state
