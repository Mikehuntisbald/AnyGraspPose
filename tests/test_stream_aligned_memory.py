import torch
import pytest
from lip.models.stream_spatial_memory import SpatialRKTracker
from lip.models.stream_aligned_memory import AlignedRKTracker,AlignedCache
from lip.models.aligned_memory import pool_memory_geometry
from test_stream_rk import features,meta


def test_nonzero_spatial_parent_is_preserved_and_source_geometry_is_immutable():
    torch.manual_seed(15);parent=SpatialRKTracker(dense_side=2).eval()
    for p in (parent.head[-1].weight,parent.patch_attention.out_proj.weight,parent.anchor_attn.out_proj.weight,parent.readout.cross_attn.out_proj.weight):torch.nn.init.normal_(p,std=.02)
    parent.reliability_strength.data.fill_(.2);parent.update_strength.data.fill_(.1)
    model=AlignedRKTracker(dense_side=2,query_side=2).eval();r=model.load_state_dict(parent.state_dict(),strict=False)
    assert not r.unexpected_keys and r.missing_keys and all(k.startswith('aligned_readout.') for k in r.missing_keys)
    a=b=None;history={};f=features()
    with torch.no_grad():
        for i in range(16):
            # Encode a distinct, known geometry at each source frame.
            f=dict(f,geometry=f['geometry'].clone());f['geometry'][:,4:7]=i/100
            history[i]=pool_memory_geometry(f['geometry'],2)
            old=None if b is None else b.patch_geometry.clone()
            expected,a=parent(f,meta(i),a);actual,new_cache=model(f,meta(i),b)
            assert torch.equal(expected['latent'],actual['latent']) and torch.equal(expected['pose_centered'],actual['pose_centered'])
            if b is not None:assert torch.equal(old,b.patch_geometry)
            b=new_cache
            for lane in range(2):
                for slot in torch.nonzero(b.anchors.valid[lane],as_tuple=False).flatten():
                    frame=int(b.anchors.frame_id[lane,slot]);assert torch.equal(b.patch_geometry[lane,slot],history[frame][lane])
    assert isinstance(b,AlignedCache) and actual['aligned_update_norm'].sum()==0
    with pytest.raises(ValueError):model(f,meta(16),a)


def test_aligned_full_tracker_has_pose_gradients_and_detaches_all_memory():
    torch.manual_seed(19);model=AlignedRKTracker(dense_side=2,query_side=2).train();torch.nn.init.normal_(model.head[-1].weight,std=.02)
    f=features();cache=None
    for i in range(12):
        out,cache=model(f,meta(i),cache)
        if i==7:cache=cache.detach()
    out['pose_centered'].sum().backward()
    assert model.aligned_readout.attention.out_proj.weight.grad.abs().sum()>0
    for name,p in model.named_parameters():
        if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),name
        else:assert p.grad is None,name
    detached=cache.detach();assert isinstance(detached,AlignedCache) and detached.variant==model.variant
    assert detached.patch_geometry.grad_fn is None and detached.patches.grad_fn is None
    assert all(b.key.grad_fn is None and b.value.grad_fn is None for layer in detached.layers for b in layer)


def test_online_aligned_cache_correction_and_reset():
    from test_stream_geometry import fixture
    mesh,pose,k=fixture();model=AlignedRKTracker(dense_side=2,query_side=2).eval();state=model.initialize(pose,mesh,k,'aligned',0.,image_shape=(64,64))
    for i in range(1,12):
        proposal,next_state=model.step(torch.zeros(3,64,64),torch.zeros(1,64,64),i/30,state,image_size=32)
        assert proposal['status']=='ok';state=model.commit(proposal,next_state)
    assert isinstance(state.cache,AlignedCache) and state.cache.patch_geometry is not None
    before=state.cache.patch_geometry.clone();fixed=model.correct(state,pose)
    assert fixed.cache is state.cache and torch.equal(fixed.cache.patch_geometry,before)
    fresh=model.correct(state,pose,relocalization=True)
    assert not fresh.cache.metadata and getattr(fresh.cache,'patch_geometry',None) is None


def test_aligned_checkpoint_cannot_be_loaded_with_changed_coordinate_bandwidth(tmp_path):
    from lip.engine.stream_checkpoint import load_init
    model=AlignedRKTracker(dense_side=2,query_side=2);config=dict(memory_frames=8,observation_reliability=True,keyframe_memory=True,keyframe_slots=4,keyframe_min_gap=4,keyframe_max_age=64,support_tolerance=.05,spatial_memory_side=2,aligned_query_side=2,aligned_sigma=.05)
    path=tmp_path/'init.pt';audit=dict(split_hash='split',mesh_hash='mesh')
    torch.save(dict(model=model.state_dict(),architecture_id=model.architecture_id,cache_contract=model.cache_contract,config=config,**audit),path)
    with pytest.raises(ValueError,match='aligned_sigma'):load_init(path,model,audit,dict(config,aligned_sigma=.1))
    load_init(path,model,audit,config)
