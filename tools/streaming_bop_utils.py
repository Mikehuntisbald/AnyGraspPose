"""Causal stream planning and state transactions for the frozen BOP tracking audit."""
from collections import defaultdict
from dataclasses import replace
import numpy as np

METHODS = ('lip_temporal', 'lip_no_feature_history', 'fp_tracking', 'lip_fp_temporal')


def legal_pose(p):
    p = np.asarray(p)
    return (p.shape == (4, 4) and np.isfinite(p).all() and p[2, 3] > .001
            and np.allclose(p[3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-3)
            and abs(np.linalg.det(p[:3, :3])-1) < 1e-3)


def plan_streams(targets, predictions):
    """Initialize at the first legal detection, never borrow a future pose backward."""
    streams = defaultdict(list)
    for t in targets:
        if t['inst_count'] != 1:
            raise ValueError('This known-object tracker requires one instance per object identity')
        streams[t['scene_id'], t['obj_id']].append(t)
    candidates = defaultdict(list)
    for row in predictions:
        identity = row['scene_id'], row['obj_id']
        if identity in streams and legal_pose(row['pose']):
            candidates[identity].append(row)
    result = []
    for (scene, obj), rows in sorted(streams.items()):
        rows.sort(key=lambda r: r['im_id'])
        available = sorted(candidates[scene, obj], key=lambda r: (r['im_id'], -r['score']))
        init = next((r for r in available if r['im_id'] <= rows[-1]['im_id']), None)
        result.append(dict(scene_id=scene, obj_id=obj, targets=rows, initializer=init))
    return result


def expected_keys(plans):
    return {(p['scene_id'], t['im_id'], p['obj_id']) for p in plans
            if p['initializer'] is not None for t in p['targets']
            if t['im_id'] >= p['initializer']['im_id']}


def clear_feature_history(state):
    """Keep accepted pose, motion and clock; clear only visual source/context caches."""
    from lip.engine.stream_state import TemporalCache, CrossCache
    if hasattr(state.cache, 'clear_visual_history'):
        return replace(state, cache=state.cache.clear_visual_history(), source_metadata=())
    cache = TemporalCache(capacity=state.cache.source.capacity)
    return replace(state, cache=CrossCache(cache), source_metadata=())


def hold_failed_step(state, timestamp):
    """Retain the last legal pose on failure without freezing the stream clock."""
    return replace(state, previous_pose=state.pose_centered.detach(),
                   previous_timestamp=state.timestamp, timestamp=float(timestamp),
                   frame_id=state.frame_id+1)
