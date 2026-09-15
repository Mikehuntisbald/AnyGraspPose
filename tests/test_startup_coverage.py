import pytest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_startup_coverage import classify, fragments, summarize


def test_fragment_positions_exclude_initializer_and_separate_prefix():
    p=fragments(list(range(20)),2,3,5)
    assert p==dict(initializer=[2],first_update=[3],prefix=[3,4,5],supervised=[6,7,8,9,10])
    p=fragments(list(range(20)),2,0,5)
    assert p['prefix']==[] and p['supervised']==[3,4,5,6,7]
    with pytest.raises(ValueError):fragments(list(range(8)),0,3,5)


def test_unknown_is_not_clear_or_occluded_and_boundaries_are_strict():
    assert [classify(v) for v in [None,0.,.2999,.3,.4999,.5,1.]]==['unknown','severe','severe','occluded','occluded','clear','clear']
    assert summarize([None,0.,.4,1.])['visibility_lt05_fraction']==.5
    assert summarize([])['severe_fraction'] is None
    with pytest.raises(ValueError):classify(1.1)
