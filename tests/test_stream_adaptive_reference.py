from dataclasses import replace
from pathlib import Path
import sys
import pytest
import torch
from lip.models.stream_pose_reference import PoseReferenceRKTracker
from lip.models.stream_adaptive_reference import AdaptiveReferenceRKTracker,AdaptiveReferenceCache
from test_stream_rk import features,meta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def pair():
    torch.manual_seed(41);parent=PoseReferenceRKTracker(dense_side=2).eval()
    torch.nn.init.normal_(parent.head[-1].weight,std=.02)
    torch.nn.init.normal_(parent.patch_attention.out_proj.weight,std=.02)
    parent.pose_reference_feedback.readout[-1].bias.data.fill_(.3)
    model=AdaptiveReferenceRKTracker(dense_side=2).eval();loaded=model.load_state_dict(parent.state_dict(),strict=False)
    assert not loaded.unexpected_keys and all(n.startswith('reference_writer.') for n in loaded.missing_keys)
    return parent,model


def test_zero_writer_preserves_parent_exactly_and_cache_survives_visual_expiry():
    parent,model=pair();f=features();initial=f['T_base_centered'].clone();a=b=None
    with torch.no_grad():
        for i in range(18):
            before=b;old,a=parent(f,meta(i),a);new,b=model(f,meta(i),b)
            assert torch.equal(old['pose_centered'],new['pose_centered'])
            assert torch.equal(b.reference.pose,initial)
            if before is not None:assert torch.equal(before.reference.pose,initial)
            f=dict(f,T_base_centered=new['pose_centered'])
    assert b.metadata[0].frame_id.min()>0 and isinstance(b.detach(),AdaptiveReferenceCache)
    cleared=b.clear_visual_history();assert cleared.reference is b.reference and not cleared.metadata
    assert cleared.anchors is None and cleared.patches is None and not cleared.contexts
    with pytest.raises(ValueError,match='own cache'):model(f,meta(18),a)


def test_write_is_after_current_output_and_future_loss_trains_writer_without_actor_pose_gradient():
    parent,model=pair();model.train();model.reference_writer.readout[-1].bias.data.fill_(.3)
    f=features();captured=[]
    def capture(module,args,result):result.retain_grad();captured.append(result)
    hook=model.reference_writer.register_forward_hook(capture)
    expected,_=parent(f,meta(0));first,cache=model(f,meta(0));first['pose_centered'].retain_grad()
    assert torch.equal(first['pose_centered'],expected['pose_centered'])
    assert not torch.equal(cache.reference.pose,f['T_base_centered'])
    saved=cache.reference.pose.detach().clone()
    out,next_cache=model(dict(f,T_base_centered=first['pose_centered'].detach()),meta(1),cache)
    (out['pose_centered'][:,0,1].sum()+out['pose_centered'][:,0,3].sum()).backward();hook.remove()
    assert first['pose_centered'].grad is None
    assert captured[0].grad is not None and (captured[0].grad.abs().sum(0)>0).all()
    assert torch.equal(cache.reference.pose.detach(),saved)
    assert next_cache.reference.pose.grad_fn is not None and next_cache.detach().reference.pose.grad_fn is None


def test_detach_boundary_cuts_reference_write_credit():
    _,model=pair();model.train();model.reference_writer.readout[-1].bias.data.fill_(.2);f=features();gates=[]
    def capture(module,args,result):result.retain_grad();gates.append(result)
    hook=model.reference_writer.register_forward_hook(capture)
    first,cache=model(f,meta(0));out,_=model(dict(f,T_base_centered=first['pose_centered'].detach()),meta(1),cache.detach())
    out['pose_centered'].sum().backward();hook.remove();assert gates[0].grad is None


def test_new_reference_observation_timestamp_blocks_future_and_lane_identity_leaks():
    _,model=pair();f=features()
    with torch.no_grad():
        _,cache=model(f,meta(0));initial=cache.reference.pose.clone()
        f=dict(f,T_base_centered=f['T_base_centered'].clone());f['T_base_centered'][:,0,3]+=.02
        _,new=model(f,replace(meta(1),stream_tag=torch.tensor([100,1])),cache)
        assert torch.equal(new.reference.pose[0],f['T_base_centered'][0]) and torch.equal(new.reference.pose[1],initial[1])
        future=replace(cache.reference,pose=cache.reference.pose+100.,observed_timestamp=meta(20).timestamp)
        _,new=model(f,meta(2),replace(cache,reference=future))
        assert torch.equal(new.reference.pose,f['T_base_centered'])


def test_adaptive_online_correction_and_relocalization_preserve_boundaries():
    from test_stream_geometry import fixture
    from streaming_bop_utils import clear_feature_history
    mesh,pose,k=fixture();model=AdaptiveReferenceRKTracker(dense_side=2).eval()
    state=model.initialize(pose,mesh,k,'adaptive',0.,image_shape=(64,64))
    with torch.no_grad():
        for i in range(1,4):
            proposal,next_state=model.step(torch.zeros(3,64,64),torch.zeros(1,64,64),i/30,state,image_size=32)
            assert proposal['status']=='ok';state=model.commit(proposal,next_state)
    old=state.cache.reference.pose.clone();corrected_pose=pose.clone();corrected_pose[0,3]+=.01
    corrected=model.correct(state,corrected_pose)
    assert corrected.cache.source is state.cache.source and corrected.cache.patches is state.cache.patches
    assert corrected.cache.reference.observed_timestamp.item()==corrected.timestamp
    assert torch.equal(corrected.cache.reference.pose[0],corrected.pose_centered) and torch.equal(state.cache.reference.pose,old)
    cleared=clear_feature_history(corrected);assert cleared.cache.reference is corrected.cache.reference and not cleared.cache.metadata
    reset=model.correct(corrected,pose,relocalization=True);assert not reset.cache.metadata and getattr(reset.cache,'reference',None) is None


def test_adaptive_checkpoint_binds_architecture_and_write_limit(tmp_path):
    from lip.engine.stream_config import make_model
    from lip.engine.stream_checkpoint import load_init
    c=dict(architecture_id='stream_rk_adaptive_reference',memory_frames=8,dropout=0.,time_unit=1/30,max_gap_seconds=.5,
        observation_reliability=True,keyframe_memory=True,keyframe_slots=4,keyframe_min_gap=4,keyframe_max_age=64,
        support_tolerance=.05,spatial_memory_side=2,reference_write_limit=.25)
    model=make_model(c);assert isinstance(model,AdaptiveReferenceRKTracker)
    audit=dict(split_hash='split',mesh_hash='mesh');path=tmp_path/'adaptive.pt'
    torch.save(dict(model=model.state_dict(),architecture_id=model.architecture_id,cache_contract=model.cache_contract,config=c,**audit),path)
    load_init(path,model,audit,c)
    with pytest.raises(ValueError,match='write limit'):load_init(path,model,audit,dict(c,reference_write_limit=.5))
    with pytest.raises(ValueError,match='architecture_id'):load_init(path,PoseReferenceRKTracker(dense_side=2),audit,c)
