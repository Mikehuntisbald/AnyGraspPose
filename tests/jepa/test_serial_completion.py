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


def test_decoded_visible_features_receive_pose_gradients_with_real_depth_retained():
    args=fixture()
    for key in ('mid','last'):args[key].requires_grad_()
    p=pack_completion(**args,feature_source='decoded')
    torch.testing.assert_close(p['feature'][...,:384],args['mid'])
    torch.testing.assert_close(p['camera'][:,2],torch.full((1,224,224),3.))
    tokens,_,_,_=CompletionRelations()(p,args['base'])
    tokens[...,0].sum().backward()
    for key in ('mid','last'):assert args[key].grad is not None and args[key].grad.abs().sum()>0


def test_decoded_readout_has_no_direct_observed_feature_input():
    args=fixture();a=pack_completion(**args,feature_source='decoded')
    args['observed_mid']=torch.randn_like(args['observed_mid'])*1000
    args['observed_last']=torch.randn_like(args['observed_last'])*1000
    b=pack_completion(**args,feature_source='decoded')
    torch.testing.assert_close(a['feature'],b['feature'],rtol=0,atol=0)


def test_dense_measurements_preserve_pixels_without_changing_feature_routing():
    args=fixture();args['visibility'].fill_(-10)
    probability=torch.zeros_like(args['measured_depth_m']);probability[...,::2]=.9
    probability.requires_grad_();args['surface'].requires_grad_()
    packet=pack_completion(**args,measurement_probability=probability)
    torch.testing.assert_close(packet['camera'][:,2,:,::2],torch.full((1,224,112),3.))
    assert packet['measured_weight'][...,::2].min()>.89
    assert not packet['measured_weight'][...,1::2].any()
    assert not packet['completed_weight'][...,::2].any()
    assert packet['completed_weight'][...,1::2].sum()>0
    packet['camera'].sum().backward()
    assert probability.grad is None
    assert args['surface'].grad[:,3].abs().sum()>0


def test_dense_selector_cannot_create_depth_or_override_bounds():
    args=fixture();probability=torch.ones_like(args['measured_depth_m'])
    args['measured_depth_m'][...,:100]=0
    args['valid'][:,0]=False
    packet=pack_completion(**args,measurement_probability=probability)
    assert not packet['measured_weight'][...,:100].any()
    assert not packet['weight'][...,:14,:14].any()


def test_aggregate_completion_limit_and_missing_depth_fallback():
    args=fixture();prob=torch.zeros_like(args['measured_depth_m']);prob[...,:14,:14]=.9
    p=pack_completion(**args,measurement_probability=prob,completion_mass_ratio=1.)
    torch.testing.assert_close(p['completed_weight'].sum(),p['measured_weight'].sum())
    args['measured_depth_m'].zero_()
    p=pack_completion(**args,measurement_probability=prob,completion_mass_ratio=1.)
    legacy=pack_completion(**args,measurement_probability=prob)
    torch.testing.assert_close(p['completed_weight'],legacy['completed_weight'],rtol=0,atol=0)
