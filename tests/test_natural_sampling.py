import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('natural_sampling',
    Path(__file__).resolve().parents[1] / 'tools/prepare_natural_sampling.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def population():
    return [dict(object_id=obj, physical_sequence=f'{obj}/{physical}',
        stream=obj * 10 + physical, start=start,
        longest_inframe_lt03_run=12 if start == 1 and (obj == 1 or physical == 0) else 0)
        for obj in (1, 2) for physical in range(3) for start in range(3)]


def test_separate_branch_rng_preserves_objects_noise_and_uniform_draws():
    rows = population()
    control, natural, choices, eligible = module.paired_manifests(rows, 200, 42, 128000, 1., 3, 1000)
    baseline = module.paired_manifests(rows, 200, 42, 128000, 0., 3)
    assert control == baseline[0] == baseline[1]
    assert set(eligible) == {1}
    for a, b, choice in zip(control, natural, choices):
        assert a['seed'] == b['seed'] and a['stream'] // 10 == b['stream'] // 10
        assert set(a) == set(b) == {'stream', 'start', 'seed'}
        if choice['object_id'] == 1:
            assert choice['hard_selected'] and b['start'] == 1
        else:
            assert choice['fallback'] and not choice['hard_selected'] and a == b


def test_order_invariance_and_offline_construction_resume():
    from collections import Counter
    rows = population()
    full = module.paired_manifests(rows, 120, 42, 128000)
    reordered = module.paired_manifests(list(reversed(rows)), 120, 42, 128000)
    assert full[:3] == reordered[:3]
    counts = Counter()
    prefix = module.paired_manifests(rows, 40, 42, 128000, hard_counts=counts)
    resumed = module.paired_manifests(rows, 80, 42, 128040, hard_counts=counts)
    assert prefix[0] == full[0][:40] and prefix[1] == full[1][:40]
    assert full[0][40:] == resumed[0] and full[1][40:] == resumed[1]


def test_hard_repeat_cap_preserves_uniform_fallback():
    from collections import Counter
    control, natural, choices, _ = module.paired_manifests(population(), 500, 42, 0, 1., 3, 2)
    counts = Counter((row['stream'], row['start']) for row, choice in zip(natural, choices) if choice['hard_selected'])
    assert max(counts.values()) == 2 and sum(counts.values()) == 6
    assert any(choice['fallback_reason'] == 'repeat_cap' for choice in choices)
    assert all(a == b for a, b, choice in zip(control, natural, choices) if choice['fallback'])


def test_invalid_settings_rejected():
    for kwargs in (dict(fraction=-.1), dict(fraction=1.1), dict(min_physical=0)):
        with pytest.raises(ValueError):
            module.paired_manifests(population(), 10, 42, 0, **kwargs)
    with pytest.raises(ValueError):
        module.paired_manifests([], 10, 42, 0)
