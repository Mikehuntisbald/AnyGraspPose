import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_initial_pose_hold import stationary_prefix


def test_motion_then_return_does_not_become_initial_static_prefix_again():
    assert stationary_prefix([0.,1.,6.,0.],[0.,.01,.01,0.]).tolist()==[True,True,False,False]


def test_either_rotation_or_translation_ends_stationary_prefix():
    assert stationary_prefix([0.,0.,0.],[0.,.03,0.]).tolist()==[True,False,False]
    assert stationary_prefix([0.,1.,1.],[0.,.01,.01]).tolist()==[True,True,True]
