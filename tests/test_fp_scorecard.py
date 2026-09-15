import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from compare_fp_scorecard import episodes, statistic
from diagnose_occlusion_events import stable_recovery


def row(frame, visibility=.2, initialization=False, update=1, ok=True):
    return dict(stream_id='s', frame_index=frame, object_id=1, physical_sequence='q',
        visibility=visibility, initialization=initialization, updates_after_initialization=update,
        adds_005=ok, status='ok')


def test_episode_and_recovery_boundaries():
    rows = [row(0, initialization=True), row(1), row(2), row(4),
            row(5,.9),row(6,.9),row(7,.9),row(8,.9,initialization=True)]
    es = episodes(rows)
    assert [e['frames'] for e in es] == [[1,2],[4]]
    assert es[0]['censor_reason'] == 'frame_gap'
    assert es[1]['post_frames'] == [5,6,7] and not es[1]['post_complete']
    assert stable_recovery(rows[4:7]) == 3
    assert stable_recovery([row(1,.9),row(3,.9),row(4,.9)]) is None
    assert stable_recovery([row(1,.9),row(2,.9,initialization=True),row(3,.9)]) is None


def test_missing_and_unknown_visibility_split_episodes():
    es = episodes([row(0,update=None),row(1),row(2,None),row(3),row(4,.9)])
    assert [e['frames'] for e in es] == [[1],[3]]
    assert es[0]['post_frames'] == []
    assert es[0]['censor_reason'] == 'initialization_or_missing_pose_or_visibility'


def test_paired_macro_direction_and_identical_bootstrap():
    values = np.array([[0.,10.,10.],[0.,10.,10.],[80.,90.,90.]])
    result = statistic(values, np.array([1,1,2]),np.array(['a','a','b']),np.array(['a','b']),
        np.array([[1,1],[2,0],[0,2]]),['fp','residual','real_mix'])
    assert result['values'] == dict(fp=40.,residual=50.,real_mix=50.)
    assert result['comparisons']['real_mix_vs_fp']['ci95'] == [10.,10.]
    assert result['comparisons']['real_mix_vs_residual']['ci95'] == [0.,0.]
