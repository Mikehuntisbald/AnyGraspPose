import torch
from lip.unified.serial_completion import pack_completion, CompletionRelations


def fixture():
    mid=torch.randn(1,256,384);last=mid.clone()
    surface=torch.zeros(1,5,224,224);surface[:,4]=5
    geometry=torch.zeros(1,9,224,224)
    rays=torch.zeros(1,3,224,224);rays[:,2]=1
    base=torch.eye(4)[None];base[:,2,3]=1
    return dict(mid=mid,last=last,surface=surface,geometry=geometry,rays=rays,base=base,
        diameter=torch.tensor([.1]),visibility=torch.full((1,256),10.),support=torch.full((1,256),10.),
        observed_mid=mid+2,observed_last=last+3,valid=torch.ones(1,256,dtype=torch.bool),
        measured_depth_m=torch.full((1,1,224,224),1.3))


def test_measured_depth_exact_and_completion_disjoint():
    args=fixture();p=pack_completion(**args)
    # Geometry encoder clipped depth at 2d; the readout must use the raw 3d residual.
    torch.testing.assert_close(p['camera'][:,2],torch.full((1,224,224),3.))
    assert not p['completed_weight'].any()
    torch.testing.assert_close(p['feature'][...,:384],args['observed_mid'])


def test_missing_depth_never_fabricates_measurement():
    args=fixture();args['measured_depth_m'].zero_();p=pack_completion(**args)
    assert not p['measured_weight'].any() and p['completed_weight'].sum()>0


def test_hidden_features_and_xyz_depth_receive_pose_gradients():
    args=fixture();args['visibility'].fill_(-10)
    for k in ('surface','mid','last'):args[k].requires_grad_()
    p=pack_completion(**args);module=CompletionRelations()
    tokens,_,moments,_=module(p,args['base']);loss=tokens[...,0].sum()+moments.sum();loss.backward()
    for k in ('surface','mid','last'):
        assert args[k].grad is not None and torch.isfinite(args[k].grad).all() and args[k].grad.abs().sum()>0
    assert args['surface'].grad[:,:3].abs().sum()>0 and args['surface'].grad[:,3].abs().sum()>0


def test_optical_rotation_has_signed_relation_signal():
    from lip.geometry.so3 import exp
    args=fixture();p=pack_completion(**args)
    xy=torch.linspace(-.3,.3,224);y,x=torch.meshgrid(xy,xy,indexing='ij')
    p['xyz']=torch.stack((x,y,torch.zeros_like(x)))[None];p['camera']=p['xyz'].clone()
    module=CompletionRelations();base=args['base'];_,_,correct,correct_scale=module(p,base)
    base=base.clone();base[:,:3,:3]=exp(torch.tensor([[0.,0.,.1745329]]))
    _,_,wrong,wrong_scale=module(p,base)
    assert correct[:,9:18].abs().max()<1e-5
    assert wrong[:,9:18].abs().max()>.05
    assert correct_scale.max()<1e-4 and wrong_scale.min()>.1


def test_no_evidence_tokens_invalid_and_finite():
    args=fixture();args['valid'].zero_();p=pack_completion(**args)
    tokens,valid,moments,_=CompletionRelations()(p,args['base'])
    assert not valid.any() and tokens.isfinite().all() and moments.isfinite().all()
