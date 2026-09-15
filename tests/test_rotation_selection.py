import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from select_rotation_candidate import selection_checks


def evidence():
    def metric(delta,ci):return {'comparisons':{'rotation_anchor_vs_control':{'delta':delta,'ci95':ci}}}
    return dict(comparison={'populations':{
        'visibility_lt_03':{'metrics':{'adds_005':metric(1.,[.1,1.9])}},
        'all':{'metrics':{'add_01':metric(.2,[-.1,.5]),'adds_005':metric(.1,[-.1,.3])}}}},
        bad_initial={'populations':{'first8':{'metrics':{'rotation_deg':{'delta':-.1},'center_mm':{'delta':-.01}}}}},
        robustness={'populations':{'initial_bad':{'metrics':{'adds_005':{'rotation_anchor':{'delta':.1}}}}}},
        fixed={'populations':{'fixed_parent_moving_lip_benefit':{'metrics':{
            'rotation_anchor':{'object_macro_percent':99.},'control':{'object_macro_percent':98.}}}}})


def test_occlusion_gain_does_not_override_startup_or_overall_regression():
    values=evidence();assert all(selection_checks(**values).values())
    bad=copy.deepcopy(values);bad['bad_initial']['populations']['first8']['metrics']['rotation_deg']['delta']=.2
    assert not selection_checks(**bad)['bad_initial_first8_rotation_point_not_worse']
    bad=copy.deepcopy(values);bad['comparison']['populations']['all']['metrics']['add_01']['comparisons']['rotation_anchor_vs_control']['delta']=-.01
    assert not selection_checks(**bad)['overall_add_point_not_lower']
    bad=copy.deepcopy(values);bad['comparison']['populations']['visibility_lt_03']['metrics']['adds_005']['comparisons']['rotation_anchor_vs_control']['ci95'][0]=-.1
    assert not selection_checks(**bad)['severe_strict_paired_ci_positive']


def test_invalid_evidence_cannot_select_a_new_checkpoint():
    values=evidence();values['bad_initial']['populations']['first8']['metrics']['center_mm']['delta']=float('nan')
    with pytest.raises(ValueError):selection_checks(**values)
    values=evidence();del values['fixed']['populations']['fixed_parent_moving_lip_benefit']
    with pytest.raises(KeyError):selection_checks(**values)
