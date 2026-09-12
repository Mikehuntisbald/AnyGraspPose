import copy
import inspect
import torch
import pytest
from test_stream_geometry import fixture
from lip.models.stream_tracker import StreamTracker
from lip.engine.stream_training import StreamTrainingModule
from lip.engine.stream_features import build_current_features
from lip.geometry.renderer import Renderer


def sample():
    mesh,t,k=fixture()
    return dict(initial_pose=t,targets=t[None].repeat(4,1,1),rgb=torch.rand(4,3,64,64),
        depth=torch.full((4,1,64,64),.6),timestamps=torch.tensor([0.,.025,.060,.10,.13],dtype=torch.float64),
        frames=torch.arange(5),mesh=mesh,k=k,sample={'seed':42},depth_scale=1.)


@pytest.mark.parametrize('batched',[False,True])
def test_gt_and_future_rgbd_do_not_enter_past_prediction(batched):
    c=dict(burn_in_frames=1,supervised_unroll_frames=3,initial_pose_noise=False,image_size=32,crop_expansion=2.,precision='fp32',batch_current_features=batched)
    m=StreamTracker(dropout=0.).eval();torch.nn.init.normal_(m.head[-1].weight,std=.001)
    runner=StreamTrainingModule(m,c,Renderer('cpu'));s=sample();features=[]
    handle=m.register_forward_pre_hook(lambda m,a:features.append({k:v.detach().clone() for k,v in a[0].items()}))
    with torch.no_grad():
        a=runner([s],True);other=copy.deepcopy(s);other['targets'][:,:3,3]+=.3
        b=runner([other],True)
    assert torch.equal(a['predictions'],b['predictions'])
    assert not torch.equal(a['loss'],b['loss'])
    for i in range(4):
        for k in features[i]:assert torch.equal(features[i][k],features[i+4][k])
    changed=copy.deepcopy(s);changed['rgb'][2:]*=0;changed['depth'][2:]*=0;changed['timestamps'][3:]+=.1
    with torch.no_grad():d=runner([changed],True)
    assert torch.equal(a['predictions'][:,:2],d['predictions'][:,:2])
    handle.remove()
    assert not any(x in inspect.signature(build_current_features).parameters for x in ('gt','target','labels'))


def test_dual_context_does_not_change_source_cache_and_gate_not_confidence():
    from lip.models.stream_readout import StreamReadout
    r=StreamReadout(True);z=torch.randn(2,2,256);out=r(z)
    assert torch.equal(out['latent'],z[:,0])
    torch.testing.assert_close(out['context_gate'],torch.full((2,1),torch.sigmoid(torch.tensor(-2.)).item()))
    assert 'confidence' not in out
    # The distinct object key bias intentionally makes dual different from single.
    from lip.models.stream_attention import StreamTemporal
    from test_stream_attention import metadata
    m=StreamTemporal(8,0.).eval();meta=metadata(1,1)[0];s=torch.randn(1,17,256);q=torch.randn(1,2,256)
    with torch.no_grad():
        a,ca=m(s,q,meta,object_readout=True);q[:,1]+=30
        b,cb=m(s,q,meta,object_readout=True)
    torch.testing.assert_close(a[:,0],b[:,0],atol=1e-6,rtol=1e-6)
    for la,lb in zip(ca.layers,cb.layers):assert torch.equal(la[0].key,lb[0].key)


@pytest.mark.parametrize('batched',[False,True])
def test_last_loss_reaches_earlier_rgb_encoder_through_kv_only(batched):
    c=dict(burn_in_frames=1,supervised_unroll_frames=3,initial_pose_noise=False,image_size=32,crop_expansion=2.,precision='fp32',batch_current_features=batched)
    m=StreamTracker(dropout=0.).train();torch.nn.init.normal_(m.head[-1].weight,std=.001)
    original=m.encode_current;sources=[]
    def encode(features,profiler=None):
        source=original(features,profiler)
        # Remove the last frame's encoder path. Any RGB parameter gradient from
        # the final pose must now travel through earlier-frame cached K/V.
        if len(sources)==3:source=source.detach()
        if source.requires_grad:source.retain_grad()
        sources.append(source);return source
    m.encode_current=encode
    result=StreamTrainingModule(m,c,Renderer('cpu'))([sample()],True)
    result['predictions'][:,-1,:3,3].sum().backward()
    assert not sources[0].requires_grad and not sources[-1].requires_grad
    assert sources[1].grad.abs().sum()>0
    assert m.rgb[0].weight.grad is not None and m.rgb[0].weight.grad.abs().sum()>0
