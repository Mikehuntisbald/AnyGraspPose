import copy
import pytest
import torch
from test_stream_geometry import fixture
from test_stream_causality import sample
from lip.engine.stream_features import build_current_features,stack_current,mesh_to_device
from lip.engine.stream_batch_features import build_current_batch
from lip.engine.stream_training import StreamTrainingModule
from lip.geometry.renderer import Renderer
from lip.models.stream_tracker import StreamTracker


def compare_features(device):
    torch.manual_seed(102)
    mesh,base,k=fixture();meshes=[mesh_to_device(mesh,device) for _ in range(4)]
    base=base.to(device)[None].repeat(4,1,1);k=k.to(device)[None].repeat(4,1,1)
    base[1,0,3]=.05;base[2,2,3]=.8;base[3,2,3]=-1
    previous=base.clone();previous[:,0,3]-=.015
    rgb=torch.rand(4,3,64,64,device=device);depth=torch.full((4,1,64,64),.6,device=device)
    depth[1]=float('nan');depth[2]=0
    now=torch.tensor([.12,.16,.23,.27],device=device,dtype=torch.float64)
    last=now-.04;past=last-.03;past[1]=last[1]
    renderer=Renderer(device);expected=[];diags=[]
    for lane in range(4):
        f,d=build_current_features(rgb[lane],depth[lane],base[lane],k[lane],meshes[lane],renderer,
                                  now[lane],last[lane],previous[lane],past[lane],size=32)
        expected.append(f);diags.append(d)
    actual,diag=build_current_batch(rgb,depth,base,k,meshes,renderer,now,last,previous,past,size=32)
    errors={}
    for name,value in stack_current(expected).items():
        errors[name]=float((value-actual[name]).abs().max())
        torch.testing.assert_close(actual[name],value,atol=3e-5,rtol=1e-5)
    for name in ('A','K_crop','role_bias','geometry_reliable','depth_valid_fraction','depth_had_nonfinite'):
        torch.testing.assert_close(diag[name],torch.stack([d[name] for d in diags]),atol=3e-5,rtol=1e-5)
    assert diag['render_calls']==4 and diag['encoded_images_requested']==4
    print('batch_feature_errors',device,errors)
    rgb[0,0,0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        build_current_batch(rgb,depth,base,k,meshes,renderer,now,last,size=32)


def test_batch_features_match_lane_reference():
    compare_features('cpu')


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_batch_features_match_lane_reference():
    compare_features('cuda')


def test_batch_training_predictions_and_gradients_match():
    torch.manual_seed(102)
    config=dict(burn_in_frames=1,supervised_unroll_frames=3,initial_pose_noise=False,
                image_size=32,crop_expansion=2.,precision='fp32')
    first=StreamTracker(dropout=0.).train();torch.nn.init.normal_(first.head[-1].weight,std=.001)
    second=copy.deepcopy(first);samples=[sample(),sample()];samples[1]['initial_pose'][0,3]=.015
    a=StreamTrainingModule(first,config,Renderer('cpu'))(samples,True)
    b=StreamTrainingModule(second,dict(config,batch_current_features=True),Renderer('cpu'))(samples,True)
    torch.testing.assert_close(a['predictions'],b['predictions'],atol=3e-5,rtol=1e-5)
    torch.testing.assert_close(a['loss'],b['loss'],atol=3e-6,rtol=1e-5)
    a['loss'].backward();b['loss'].backward()
    largest=0.
    for (name,x),(other,y) in zip(first.named_parameters(),second.named_parameters()):
        assert name==other
        if x.grad is None:assert y.grad is None;continue
        largest=max(largest,float((x.grad-y.grad).abs().max()))
        torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=3e-4)
    print('batch_training_gradient_max_abs',largest)
