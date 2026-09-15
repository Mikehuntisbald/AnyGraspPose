import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from streaming_bop_utils import plan_streams, expected_keys, legal_pose, clear_feature_history, hold_failed_step


def row(scene, frame, obj, score=1.):
    pose = np.eye(4)
    pose[2, 3] = .7
    return dict(scene_id=scene, im_id=frame, obj_id=obj, score=score, pose=pose)


def test_causal_initialization_does_not_fill_past_with_future_detection():
    targets = [dict(scene_id=1, im_id=i, obj_id=2, inst_count=1) for i in (0, 4, 8)]
    low, high = row(1, 4, 2, .1), row(1, 4, 2, .9)
    plans = plan_streams(targets, [row(1, 8, 2), low, high, row(2, 0, 2)])
    assert plans[0]['initializer'] is high
    assert expected_keys(plans) == {(1, 4, 2), (1, 8, 2)}


def test_missing_and_invalid_initializers_remain_missing():
    targets = [dict(scene_id=1, im_id=0, obj_id=2, inst_count=1)]
    bad = row(1, 0, 2)
    bad['pose'][0, 0] = -1
    assert not legal_pose(bad['pose'])
    assert expected_keys(plan_streams(targets, [bad, row(1, 4, 2)])) == set()
    with pytest.raises(ValueError):
        plan_streams([dict(targets[0], inst_count=2)], [])


def test_cache_ablation_preserves_motion_and_failed_clock_can_advance():
    from lip.models.stream_tracker import StreamTracker
    from lip.geometry.renderer import Renderer
    from test_stream_geometry import fixture
    mesh, pose, K = fixture()
    model = StreamTracker(architecture_id='stream_dual_cross_residual', dropout=0.).eval()
    state = model.initialize(pose, mesh, K, 'one', 0., image_shape=(64, 64))
    with torch.no_grad():
        prediction, pending = model.step(torch.zeros(3, 64, 64), torch.zeros(1, 64, 64), 1/30,
                                         state, renderer=Renderer('cpu'), image_size=32)
        state = model.commit(prediction, pending)
    assert len(state.cache.metadata) == 1
    cleared = clear_feature_history(state)
    assert not cleared.cache.metadata and not cleared.cache.contexts and not cleared.source_metadata
    assert cleared.pose_centered is state.pose_centered
    assert cleared.previous_pose is state.previous_pose and cleared.previous_timestamp == state.previous_timestamp
    assert cleared.timestamp == state.timestamp and cleared.frame_id == state.frame_id
    held = hold_failed_step(state, 2/30)
    assert held.pose_centered is state.pose_centered and held.cache is state.cache
    assert held.timestamp == 2/30 and held.previous_timestamp == state.timestamp
    corrected = model.correct(state, prediction['pose_original'])
    assert corrected.cache is state.cache and corrected.source_metadata is state.source_metadata
