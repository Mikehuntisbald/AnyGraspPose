from dataclasses import replace
from pathlib import Path
import sys
import torch
import pytest
from lip.geometry.so3 import exp
from lip.models.stream_adaptive_reference import AdaptiveReferenceRKTracker,AdaptiveReferenceCache
from lip.models.stream_rotation_anchor import RotationAnchorRKTracker,RotationAnchorCache
from test_stream_rk import features,meta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def pair():
    torch.manual_seed(41);parent=AdaptiveReferenceRKTracker(dense_side=2).eval()
    torch.nn.init.normal_(parent.head[-1].weight,std=.02)
    torch.nn.init.normal_(parent.patch_attention.out_proj.weight,std=.02)
    parent.pose_reference_feedback.readout[-1].bias.data.fill_(.3)
    parent.reference_writer.readout[-1].bias.data.copy_(torch.tensor([.35,.3]))
    model=RotationAnchorRKTracker(dense_side=2).eval()
    loaded=model.load_state_dict(parent.state_dict(),strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('rotation_anchor_readout.') for k in loaded.missing_keys)
    return parent,model


def test_zero_anchor_matches_trained_adaptive_closed_loop_and_retains_initial_rotation():
    parent,model=pair();f=features();initial=f['T_base_centered'][:,:3,:3].clone();a=b=None
    with torch.no_grad():
        for i in range(18):
            old,a=parent(f,meta(i),a);new,b=model(f,meta(i),b)
            assert torch.equal(old['pose_centered'],new['pose_centered'])
            assert torch.equal(a.reference.pose,b.reference.pose)
            assert torch.equal(b.initial_rotation,initial)
            f=dict(f,T_base_centered=new['pose_centered'])
    assert not torch.equal(b.reference.pose[:,:3,:3],initial)
    assert isinstance(b.detach(),RotationAnchorCache)
    cleared=b.clear_visual_history()
    assert cleared.reference is b.reference and cleared.initial_rotation is b.initial_rotation
    assert not cleared.metadata and cleared.patches is None
    with pytest.raises(ValueError,match='own cache'):model(f,meta(18),a)


def test_rotation_read_does_not_change_center_or_writer_at_identical_input_state():
    _,model=pair();f=features()
    with torch.no_grad():
        _,cache=model(f,meta(0))
        shifted=cache.reference.pose.clone();shifted[:,:3,:3]=exp(torch.tensor([[.2,.3,.1],[-.2,.1,.3]]))
        cache=replace(cache,reference=replace(cache.reference,pose=shifted))
        zero,z=model(f,meta(1),cache)
        model.rotation_anchor_readout.readout[-1].bias.data.fill_(1.)
        anchor,a=model(f,meta(1),cache)
    assert not torch.equal(zero['pose_centered'][:,:3,:3],anchor['pose_centered'][:,:3,:3])
    assert torch.equal(zero['pose_centered'][:,:3,3],anchor['pose_centered'][:,:3,3])
    assert torch.equal(z.reference.pose,a.reference.pose)
    assert torch.equal(zero['reference_center_coefficient'],anchor['reference_center_coefficient'])


def test_current_anchor_credit_future_writer_credit_and_actor_detach():
    _,model=pair();model.train();f=features();writers=[];reads=[]
    def keep(collection):
        def hook(module,args,result):result.retain_grad();collection.append(result)
        return hook
    hooks=[model.reference_writer.register_forward_hook(keep(writers)),model.rotation_anchor_readout.register_forward_hook(keep(reads))]
    first,cache=model(f,meta(0));first['pose_centered'].retain_grad()
    second,cache2=model(dict(f,T_base_centered=first['pose_centered'].detach()),meta(1),cache)
    (second['pose_centered'][:,0,1].sum()+second['pose_centered'][:,0,3].sum()).backward()
    for h in hooks:h.remove()
    assert first['pose_centered'].grad is None
    assert writers[0].grad is not None and (writers[0].grad.abs().sum(0)>0).all()
    assert reads[1].grad is not None and reads[1].grad.abs().sum()>0
    assert model.rotation_anchor_readout.readout[-1].weight.grad.abs().sum()>0
    assert not cache2.initial_rotation.requires_grad
    assert cache2.reference.pose.grad_fn is not None and cache2.detach().reference.pose.grad_fn is None


def test_rotation_anchor_and_reference_both_reset_on_future_or_changed_lane():
    _,model=pair();f=features()
    with torch.no_grad():
        _,cache=model(f,meta(0));original=cache.initial_rotation.clone()
        changed=f['T_base_centered'].clone();changed[:,:3,:3]=exp(torch.tensor([[.3,0.,0.],[0.,.2,0.]]))
        f=dict(f,T_base_centered=changed)
        _,next_cache=model(f,replace(meta(1),stream_tag=torch.tensor([100,1])),cache)
        assert torch.equal(next_cache.initial_rotation[0],changed[0,:3,:3])
        assert torch.equal(next_cache.initial_rotation[1],original[1])
        future=replace(cache.reference,observed_timestamp=meta(20).timestamp)
        _,fresh=model(f,meta(2),replace(cache,reference=future))
        assert torch.equal(fresh.initial_rotation,changed[:,:3,:3])


def test_rotation_anchor_cache_detach_cuts_previous_writer_credit():
    _,model=pair();model.train();f=features();writes=[]
    def capture(module,args,result):result.retain_grad();writes.append(result)
    hook=model.reference_writer.register_forward_hook(capture)
    first,cache=model(f,meta(0))
    second,_=model(dict(f,T_base_centered=first['pose_centered'].detach()),meta(1),cache.detach())
    second['pose_centered'].sum().backward();hook.remove()
    assert writes[0].grad is None


def test_rotation_anchor_online_correction_history_clear_and_relocalization():
    from test_stream_geometry import fixture
    from streaming_bop_utils import clear_feature_history
    mesh,pose,k=fixture();model=RotationAnchorRKTracker(dense_side=2).eval()
    state=model.initialize(pose,mesh,k,'rotation-anchor',0.,image_shape=(64,64))
    with torch.no_grad():
        for i in range(1,4):
            proposal,next_state=model.step(torch.zeros(3,64,64),torch.zeros(1,64,64),i/30,state,image_size=32)
            assert proposal['status']=='ok';state=model.commit(proposal,next_state)
    old=state.cache.initial_rotation.clone();corrected_pose=pose.clone()
    corrected_pose[:3,:3]=exp(torch.tensor([.1,.2,0.]))
    corrected=model.correct(state,corrected_pose)
    assert corrected.cache.source is state.cache.source
    assert torch.equal(corrected.cache.initial_rotation,corrected.pose_centered[None,:3,:3])
    assert torch.equal(state.cache.initial_rotation,old)
    cleared=clear_feature_history(corrected)
    assert cleared.cache.initial_rotation is corrected.cache.initial_rotation and not cleared.cache.metadata
    reset=model.correct(corrected,pose,relocalization=True)
    assert not reset.cache.metadata and getattr(reset.cache,'initial_rotation',None) is None


def test_rotation_anchor_checkpoint_contract_and_write_limit(tmp_path):
    from lip.engine.stream_config import make_model
    from lip.engine.stream_checkpoint import load_init
    c=dict(architecture_id='stream_rk_rotation_anchor',memory_frames=8,dropout=0.,time_unit=1/30,max_gap_seconds=.5,
        observation_reliability=True,keyframe_memory=True,keyframe_slots=4,keyframe_min_gap=4,keyframe_max_age=64,
        support_tolerance=.05,spatial_memory_side=2,reference_write_limit=.25)
    model=make_model(c);assert isinstance(model,RotationAnchorRKTracker)
    audit=dict(split_hash='split',mesh_hash='mesh');path=tmp_path/'rotation_anchor.pt'
    torch.save(dict(model=model.state_dict(),architecture_id=model.architecture_id,cache_contract=model.cache_contract,config=c,**audit),path)
    load_init(path,model,audit,c)
    with pytest.raises(ValueError,match='write limit'):load_init(path,model,audit,dict(c,reference_write_limit=.5))
    with pytest.raises(ValueError,match='architecture_id'):load_init(path,AdaptiveReferenceRKTracker(dense_side=2),audit,c)
