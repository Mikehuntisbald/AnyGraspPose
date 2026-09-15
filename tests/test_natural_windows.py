import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('natural_windows',Path(__file__).resolve().parents[1]/'tools/analyze_natural_windows.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_retention_and_recovery_require_clear_observations_and_complete_tail():
    v=[.9]*8+[.2]*20+[.9]*28;d=module.describe_window(v,[True]*56)
    assert d['natural_lt03_frames']==20 and d['longest_lt03_run']==20 and d['retention_ready'] and d['recovery_ready']
    v[-32:]=[.2]*32;d=module.describe_window(v,[True]*56)
    assert not d['retention_ready'] and not d['recovery_ready']


def test_unknown_visibility_splits_runs_and_burn_in_is_not_supervision():
    v=[.2]*8+[.2]*4+[None]+[.2]*4+[.9]*39;d=module.describe_window(v,[True]*56)
    assert d['natural_lt03_frames']==8 and d['unknown_frames']==1 and d['longest_lt03_run']==4
    assert not d['retention_ready'] and not d['recovery_ready']


def test_boundary_crossing_does_not_qualify_as_inframe_retention():
    v=[.9]*8+[.2]*12+[.9]*36;inside=[True]*56;inside[10]=False
    d=module.describe_window(v,inside);assert not d['retention_ready'] and not d['recovery_ready']
    assert d['natural_lt03_frames']==12
    with pytest.raises(ValueError):module.describe_window(v[:-1],inside)
