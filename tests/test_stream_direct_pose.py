import torch
import pytest
from lip.models.stream_spatial_memory import SpatialRKTracker,SpatialCache
from lip.models.stream_direct_pose import DirectPoseRKTracker
from test_stream_rk import features,meta


def test_direct_tracker_preserves_trained_parent_and_rejects_other_cache():
    torch.manual_seed(11);parent=SpatialRKTracker(dense_side=2).eval()
    for parameter in (parent.head[-1].weight,parent.patch_attention.out_proj.weight,parent.anchor_attn.out_proj.weight,parent.readout.cross_attn.out_proj.weight):
        torch.nn.init.normal_(parameter,std=.02)
    parent.update_strength.data.fill_(.1)
    model=DirectPoseRKTracker(dense_side=2).eval();loaded=model.load_state_dict(parent.state_dict(),strict=False)
    assert not loaded.unexpected_keys and loaded.missing_keys and all(k.startswith('direct_pose_residual.') for k in loaded.missing_keys)
    f=features();a=b=None
    with torch.no_grad():
        for i in range(14):
            before=None if b is None else b.patches.clone()
            expected,a=parent(f,meta(i),a);actual,next_cache=model(f,meta(i),b)
            for key in ('latent','pose_centered','pose_original','delta_rotvec','delta_center_norm'):
                assert torch.equal(expected[key],actual[key]),key
            if b is not None:assert torch.equal(before,b.patches)
            b=next_cache;f=dict(f,T_base_centered=actual['pose_centered'])
    assert isinstance(b,SpatialCache) and b.variant==model.variant
    with pytest.raises(ValueError,match='configuration'):model(f,meta(14),a)


def test_direct_tracker_backpropagates_and_detaches_memory():
    torch.manual_seed(8);model=DirectPoseRKTracker(dense_side=2).train()
    torch.nn.init.normal_(model.head[-1].weight,std=.02)
    f=features();cache=None
    for i in range(12):
        out,cache=model(f,meta(i),cache)
        if i==7:cache=cache.detach()
    loss=out['pose_centered'][:,0,1].sum()+out['pose_centered'][:,0,3].sum();loss.backward()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
        else:assert parameter.grad is None,name
    for head in (model.direct_pose_residual.rotation,model.direct_pose_residual.center):
        assert head[-1].weight.grad.abs().sum()>0
    detached=cache.detach()
    assert detached.patches.grad_fn is None and detached.anchors.latent.grad_fn is None
    assert all(block.key.grad_fn is None for layer in detached.layers for block in layer)


def test_online_direct_pose_correction_and_relocalization():
    from test_stream_geometry import fixture
    mesh,pose,k=fixture();model=DirectPoseRKTracker(dense_side=2).eval()
    model.direct_pose_residual.center[-1].bias.data.fill_(.001)
    state=model.initialize(pose,mesh,k,'direct',0.,image_shape=(64,64))
    for i in range(1,12):
        proposal,next_state=model.step(torch.zeros(3,64,64),torch.zeros(1,64,64),i/30,state,image_size=32)
        assert proposal['status']=='ok';state=model.commit(proposal,next_state)
    assert isinstance(state.cache,SpatialCache) and state.cache.variant==model.variant
    fixed=model.correct(state,pose);assert fixed.cache is state.cache
    fresh=model.correct(state,pose,relocalization=True)
    assert not fresh.cache.metadata and fresh.cache_contract==model.cache_contract


def test_direct_checkpoint_identity_and_factory(tmp_path):
    from lip.engine.stream_checkpoint import load_init
    from lip.engine.stream_config import make_model
    config=dict(architecture_id='stream_rk_direct_pose',memory_frames=8,dropout=0.,time_unit=1/30,max_gap_seconds=.5,
        observation_reliability=True,keyframe_memory=True,keyframe_slots=4,keyframe_min_gap=4,keyframe_max_age=64,support_tolerance=.05,spatial_memory_side=2)
    model=make_model(config);assert isinstance(model,DirectPoseRKTracker)
    audit=dict(split_hash='split',mesh_hash='mesh');path=tmp_path/'direct.pt'
    torch.save(dict(model=model.state_dict(),architecture_id=model.architecture_id,cache_contract=model.cache_contract,config=config,**audit),path)
    load_init(path,model,audit,config)
    with pytest.raises(ValueError,match='architecture_id'):load_init(path,SpatialRKTracker(dense_side=2),audit,config)
