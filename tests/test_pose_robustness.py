from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from analyze_pose_robustness import initial_groups,stable_success_position,effect


def row(frame,success=True,status='ok',initial=False):
    return dict(stream_id='a/b/c',object_id=1,frame_index=frame,initialization=initial,
        adds_005=success,adds_01=success,status=status)


def test_initial_quality_uses_only_initialization_not_later_recovery():
    rows=[row(0,False,initial=True),row(1),row(2),row(3)]
    assert initial_groups(rows)['a/b/c']['good'] is False
    assert initial_groups(rows)['a/b/c']['very_bad'] is True
    assert stable_success_position(rows)==3


def test_stable_recovery_excludes_init_invalid_steps_and_frame_gaps():
    rows=[row(0,initial=True),row(1),row(2),row(3,status='invalid_proposal_pose'),row(4),row(6),row(7),row(8)]
    assert stable_success_position(rows)==7
    assert stable_success_position(rows[:3]) is None


def test_macro_weighting_and_insufficient_cluster_interval():
    values=np.array([[100.,0.],[100.,0.],[0.,100.]])
    summaries=effect(values,np.array([1,1,2]),np.array(['a','a','b']),np.array([[1,1],[2,0],[0,2]]),0)
    assert summaries[0]['value']==summaries[1]['value']==50 and summaries[1]['delta']==0
    single=effect(values[:2],np.array([1,1]),np.array(['a','a']),np.ones((3,1)),0)
    assert single[1]['delta']==-100 and single[1]['ci95'] is None
