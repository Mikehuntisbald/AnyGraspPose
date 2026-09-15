from dataclasses import replace
import torch
import pytest
from lip.models.smooth_rotation_anchor import signed_anchor_coefficient,SmoothRotationAnchorReadout,SmoothRotationAnchorRKTracker
from lip.models.stream_rotation_anchor import RotationAnchorRKTracker
from lip.models.stream_adaptive_reference import AdaptiveReferenceRKTracker
from lip.geometry.so3 import exp
from test_stream_rk import features,meta


def test_softsign_zero_and_negative_values_keep_gradients_without_hard_intervals():
    x=torch.tensor([-100.,-1.,-.001,0.,.001,1.,100.],requires_grad=True)
    y=signed_anchor_coefficient(x);y.sum().backward()
    assert y[3]==0 and (y.abs()<1).all() and (x.grad>0).all()
    torch.testing.assert_close(x.grad,1/(1+x.detach().abs()).square(),rtol=1e-4,atol=1e-8)


def test_negative_bias_can_recover_to_positive_instead_of_remaining_dead():
    torch.manual_seed(7);read=SmoothRotationAnchorReadout()
    read.readout[-1].bias.data.fill_(-.01)
    inputs=(torch.randn(2,256),torch.randn(2,24),torch.randn(2,6),torch.ones(2),torch.ones(2))
    opt=torch.optim.SGD([read.readout[-1].bias],lr=.2);before=float(read(*inputs).detach().mean())
    for _ in range(4):
        opt.zero_grad();loss=(read(*inputs)-.1).square().mean();loss.backward()
        assert read.readout[-1].bias.grad.abs().sum()>0
        opt.step()
    assert before<0 and float(read(*inputs).detach().mean())>0


def pair():
    torch.manual_seed(41);parent=AdaptiveReferenceRKTracker(dense_side=2).eval()
    torch.nn.init.normal_(parent.head[-1].weight,std=.02)
    torch.nn.init.normal_(parent.patch_attention.out_proj.weight,std=.02)
    parent.pose_reference_feedback.readout[-1].bias.data.fill_(.3)
    parent.reference_writer.readout[-1].bias.data.copy_(torch.tensor([.35,.3]))
    model=SmoothRotationAnchorRKTracker(dense_side=2).eval()
    missing=model.load_state_dict(parent.state_dict(),strict=False)
    assert not missing.unexpected_keys and all(k.startswith('rotation_anchor_readout.') for k in missing.missing_keys)
    return parent,model


def test_smooth_zero_start_is_exact_closed_loop_parent_and_cache_is_separate():
    parent,model=pair();f=features();a=b=None
    with torch.no_grad():
        for i in range(18):
            old,a=parent(f,meta(i),a);new,b=model(f,meta(i),b)
            assert torch.equal(old['pose_centered'],new['pose_centered'])
            assert torch.equal(a.reference.pose,b.reference.pose)
            assert torch.count_nonzero(new['rotation_anchor_coefficient'])==0
            f=dict(f,T_base_centered=new['pose_centered'])
        with pytest.raises(ValueError,match='configuration changed'):RotationAnchorRKTracker(dense_side=2)(f,meta(18),b)
    assert b.detach().variant==model.variant and b.clear_visual_history().variant==model.variant


@pytest.mark.parametrize('bias',[-.001,1.])
def test_signed_read_changes_rotation_preserves_center_and_writing_and_receives_credit(bias):
    _,model=pair();f=features()
    with torch.no_grad():
        _,cache=model(f,meta(0));shift=cache.reference.pose.clone()
        shift[:,:3,:3]=exp(torch.tensor([[.2,.3,.1],[-.2,.1,.3]]))
        cache=replace(cache,reference=replace(cache.reference,pose=shift))
    # Keep both comparisons on the same autograd/attention execution path.
    zero,z=model(f,meta(1),cache)
    model.rotation_anchor_readout.readout[-1].bias.data.fill_(bias)
    logits=[]
    def hook(module,args,result):result.retain_grad();logits.append(result)
    h=model.rotation_anchor_readout.readout[-1].register_forward_hook(hook)
    new,c=model(f,meta(1),cache)
    assert not torch.equal(zero['pose_centered'][:,:3,:3],new['pose_centered'][:,:3,:3])
    assert torch.equal(zero['pose_centered'][:,:3,3],new['pose_centered'][:,:3,3])
    assert torch.equal(z.reference.pose,c.reference.pose)
    new['pose_centered'][:,0,1].sum().backward();h.remove()
    assert logits[0].grad.abs().sum()>0 and model.rotation_anchor_readout.readout[-1].weight.grad.abs().sum()>0


def test_smooth_and_clamped_weights_have_distinct_checkpoint_contracts(tmp_path):
    from lip.engine.stream_config import make_model
    from lip.engine.stream_checkpoint import load_init
    c=dict(architecture_id='stream_rk_rotation_anchor_smooth',memory_frames=8,dropout=0.,time_unit=1/30,max_gap_seconds=.5,
        observation_reliability=True,keyframe_memory=True,keyframe_slots=4,keyframe_min_gap=4,keyframe_max_age=64,
        support_tolerance=.05,spatial_memory_side=2,reference_write_limit=.25)
    model=make_model(c);audit=dict(split_hash='split',mesh_hash='mesh');path=tmp_path/'model.pt'
    torch.save(dict(model=model.state_dict(),architecture_id=model.architecture_id,cache_contract=model.cache_contract,config=c,**audit),path)
    load_init(path,model,audit,c)
    with pytest.raises(ValueError,match='architecture_id'):load_init(path,RotationAnchorRKTracker(dense_side=2),audit,c)
    with pytest.raises(ValueError,match='write limit'):load_init(path,model,audit,dict(c,reference_write_limit=.5))
