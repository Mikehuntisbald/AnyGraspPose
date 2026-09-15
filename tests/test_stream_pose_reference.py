from dataclasses import replace
from pathlib import Path
import sys
import pytest
import torch
from lip.models.stream_spatial_memory import SpatialRKTracker
from lip.models.stream_pose_reference import PoseReferenceRKTracker, PoseReferenceCache
from test_stream_rk import features, meta
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def test_reference_keeps_parent_trajectory_and_survives_visual_expiry():
    torch.manual_seed(12);parent = SpatialRKTracker(dense_side=2).eval()
    for name in ('head',): torch.nn.init.normal_(getattr(parent, name)[-1].weight, std=.02)
    torch.nn.init.normal_(parent.patch_attention.out_proj.weight, std=.02)
    model = PoseReferenceRKTracker(dense_side=2).eval()
    loaded = model.load_state_dict(parent.state_dict(), strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('pose_reference_feedback.') for k in loaded.missing_keys)
    f = features();initial = f['T_base_centered'].clone();a = b = None
    with torch.no_grad():
        for i in range(18):
            old = b
            expected, a = parent(f, meta(i), a);actual, b = model(f, meta(i), b)
            assert torch.equal(expected['pose_centered'], actual['pose_centered'])
            assert torch.equal(b.reference.pose, initial)
            if old is not None: assert torch.equal(old.reference.pose, initial)
            f = dict(f, T_base_centered=actual['pose_centered'])
    assert b.metadata[0].frame_id.min() > 0
    detached = b.detach();assert isinstance(detached, PoseReferenceCache) and detached.reference.pose.grad_fn is None
    empty = b.clear_visual_history()
    assert not empty.metadata and not empty.contexts and empty.anchors is None and empty.patches is None
    assert empty.reference is b.reference
    _, next_cache = model(f, meta(18), empty)
    assert torch.equal(next_cache.reference.pose, initial)
    with pytest.raises(ValueError, match='own cache'): model(f, meta(18), a)


def test_reference_identity_changes_per_lane_and_future_reference_cannot_leak():
    model = PoseReferenceRKTracker(dense_side=2).eval();f = features()
    with torch.no_grad():
        _, cache = model(f, meta(0))
        before = cache.reference.pose.clone();f['T_base_centered'] = f['T_base_centered'].clone();f['T_base_centered'][:, 0, 3] += .02
        metadata = replace(meta(1), stream_tag=torch.tensor([100, 1]))
        _, new = model(f, metadata, cache)
        assert torch.equal(new.reference.pose[0], f['T_base_centered'][0])
        assert torch.equal(new.reference.pose[1], before[1])
        future = replace(cache.reference, pose=cache.reference.pose + 100., started_timestamp=meta(50).timestamp)
        _, new = model(f, meta(2), replace(cache, reference=future))
        assert torch.equal(new.reference.pose, f['T_base_centered'])
        assert torch.equal(cache.reference.pose, before)


def test_reference_feedback_closed_loop_gradient_and_immutable_cache():
    model = PoseReferenceRKTracker(dense_side=2).train()
    torch.nn.init.normal_(model.head[-1].weight, std=.02)
    f = features();cache = None
    for i in range(12):
        out, cache = model(f, meta(i), cache)
        f = dict(f, T_base_centered=out['pose_centered'].detach())
        if i == 7: cache = cache.detach()
    (out['pose_centered'][:, 0, 1].sum() + out['pose_centered'][:, 0, 3].sum()).backward()
    for name, p in model.named_parameters():
        if p.requires_grad: assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else: assert p.grad is None, name
    assert (model.pose_reference_feedback.readout[-1].weight.grad.abs().sum(1) > 0).all()
    assert cache.reference.pose.grad_fn is None


def test_online_correction_refreshes_reference_only_and_reset_clears_it():
    from test_stream_geometry import fixture
    from streaming_bop_utils import clear_feature_history
    mesh, pose, k = fixture();model = PoseReferenceRKTracker(dense_side=2).eval()
    state = model.initialize(pose, mesh, k, 'reference', 0., image_shape=(64, 64))
    with torch.no_grad():
        for i in range(1, 12):
            proposal, candidate = model.step(torch.zeros(3, 64, 64), torch.zeros(1, 64, 64), i/30, state, image_size=32)
            assert proposal['status'] == 'ok';state = model.commit(proposal, candidate)
    prior = state.cache.reference.pose.clone()
    corrected_pose = pose.clone();corrected_pose[0, 3] += .01
    corrected = model.correct(state, corrected_pose)
    assert corrected.cache.source is state.cache.source and corrected.cache.patches is state.cache.patches
    assert corrected.cache.anchors is state.cache.anchors and corrected.cache.contexts is state.cache.contexts
    assert torch.equal(corrected.cache.reference.pose[0], corrected.pose_centered)
    assert torch.equal(state.cache.reference.pose, prior)
    ablated = clear_feature_history(corrected)
    assert ablated.cache.reference is corrected.cache.reference and not ablated.cache.metadata
    reset = model.correct(corrected, pose, relocalization=True)
    assert not reset.cache.metadata and getattr(reset.cache, 'reference', None) is None


def test_reference_checkpoint_factory_and_cache_identity(tmp_path):
    from lip.engine.stream_config import make_model
    from lip.engine.stream_checkpoint import load_init
    config = dict(architecture_id='stream_rk_pose_reference', memory_frames=8, dropout=0., time_unit=1/30,
        max_gap_seconds=.5, observation_reliability=True, keyframe_memory=True, keyframe_slots=4,
        keyframe_min_gap=4, keyframe_max_age=64, support_tolerance=.05, spatial_memory_side=2)
    model = make_model(config);assert isinstance(model, PoseReferenceRKTracker)
    audit = dict(split_hash='split', mesh_hash='mesh');path = tmp_path/'reference.pt'
    torch.save(dict(model=model.state_dict(), architecture_id=model.architecture_id,
        cache_contract=model.cache_contract, config=config, **audit), path)
    load_init(path, model, audit, config)
    with pytest.raises(ValueError, match='architecture_id'): load_init(path, SpatialRKTracker(dense_side=2), audit, config)
