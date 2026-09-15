"""Prepare matched uniform/natural-occlusion manifests; never start training.

Visibility labels select train fragments offline. Only stream/start/noise seed
enter StreamClips; no visibility, GT mask, or event label enters the predictor.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups_for(windows, hard=False):
    groups = {}
    for row in windows:
        if hard and row['longest_inframe_lt03_run'] <= 8:
            continue
        groups.setdefault(row['object_id'], {}).setdefault(
            row['physical_sequence'], {}).setdefault(row['stream'], []).append(row['start'])
    for physicals in groups.values():
        for streams in physicals.values():
            for stream in streams:
                streams[stream] = sorted(set(streams[stream]))
    return groups


def draw(groups, rng, obj):
    physicals = groups[obj]
    physical = sorted(physicals)[int(rng.integers(len(physicals)))]
    cameras = physicals[physical]
    stream = sorted(cameras)[int(rng.integers(len(cameras)))]
    starts = cameras[stream]
    return dict(stream=stream, start=starts[int(rng.integers(len(starts)))])


def paired_manifests(windows, count, seed, offset, fraction=.25, min_physical=3,
                     max_hard_repeats=8, hard_counts=None):
    if not 0 <= fraction <= 1 or min_physical < 1 or count < 1 or offset < 0 or max_hard_repeats < 1:
        raise ValueError('Invalid sampling settings')
    groups = groups_for(windows)
    hard = groups_for(windows, hard=True)
    eligible = {obj: group for obj, group in hard.items() if len(group) >= min_physical}
    objects = sorted(groups)
    if not objects:
        raise ValueError('No train windows')
    control, curriculum, selections = [], [], []
    hard_counts = Counter() if hard_counts is None else hard_counts
    for counter in range(offset, offset + count):
        # Exact StreamClips.choose RNG order, including its noise seed draw.
        rng = np.random.default_rng(seed + counter)
        obj = int(rng.choice(objects))
        item = draw(groups, rng, obj)
        item['seed'] = int(rng.integers(2**31))
        branch_rng = np.random.default_rng(np.random.SeedSequence([seed, counter, 918273]))
        requested = bool(branch_rng.random() < fraction)
        candidate = item.copy()
        chosen = False
        if requested and obj in eligible:
            # Bounded rejection within the same object; exhausted pools fall
            # back to the exact uniform draw. State is saved only in offline
            # construction; training/resume consumes the completed manifest.
            for _ in range(32):
                proposal = draw(eligible, branch_rng, obj)
                key = (proposal['stream'], proposal['start'])
                if hard_counts[key] < max_hard_repeats:
                    candidate = proposal
                    hard_counts[key] += 1
                    chosen = True
                    break
        candidate['seed'] = item['seed']
        control.append(item)
        curriculum.append(candidate)
        selections.append(dict(counter=counter, object_id=obj, requested_hard=requested,
            hard_selected=chosen, fallback=requested and not chosen,
            fallback_reason=('insufficient_physical_sequences' if obj not in eligible else 'repeat_cap')
                if requested and not chosen else None))
    return control, curriculum, selections, eligible


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('analysis', 'cache', 'index-root', 'data-root', 'runtime', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=64000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--offset', type=int, default=128000)
    parser.add_argument('--fraction', type=float, default=.25)
    parser.add_argument('--min-physical', type=int, default=3)
    parser.add_argument('--max-hard-repeats', type=int, default=8)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(args.runtime / 'src'))
    from lip.data.stream_clips import StreamClips
    from analyze_natural_windows import summary
    analysis = json.loads((args.analysis / 'analysis.json').read_text())
    cache = json.loads((args.cache / 'completed.json').read_text())
    assert cache['completed'] and cache['split'] == 'train' and cache['cached_mismatches'] == 0
    assert sha(args.cache / 'train_visibility.jsonl') == cache['output_sha256'] == analysis['visibility_cache_sha256']
    assert sha(args.analysis / 'windows.jsonl') == analysis['windows_sha256']
    windows = list(map(json.loads, (args.analysis / 'windows.jsonl').read_text().splitlines()))
    lookup = {(r['stream'], r['start']): r for r in windows}
    assert len(lookup) == len(windows)
    ds = StreamClips(args.data_root, args.index_root, 8, 48, seed=args.seed, start_sample=args.offset)
    assert ds.audit['split_hash'] == cache['split_hash'] and ds.audit['mesh_hash'] == cache['mesh_hash']
    for sid, stream in enumerate(ds.streams):
        rows = [lookup[(sid, int(start))] for start in ds.starts[sid]]
        assert all(row['stream_id'] == stream['stream_id'] and row['object_id'] == stream['object_id'] for row in rows)
    assert sum(len(v) for v in ds.starts) == len(windows)
    control, curriculum, selections, eligible = paired_manifests(
        windows, args.count, args.seed, args.offset, args.fraction, args.min_physical, args.max_hard_repeats)
    assert all(item == ds.choose(i) for i, item in enumerate(control)), 'Uniform arm must reproduce StreamClips exactly'
    assert all(lookup[(x['stream'], x['start'])]['object_id'] == lookup[(y['stream'], y['start'])]['object_id']
               and x['seed'] == y['seed'] for x, y in zip(control, curriculum))
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(completed=True, training_started=False, seed=args.seed,
        sample_counter_range=[args.offset, args.offset + args.count - 1],
        requested_hard_fraction=args.fraction, minimum_physical_sequences_per_object=args.min_physical,
        max_hard_branch_repeats_per_window=args.max_hard_repeats,
        selected_hard_fragments=sum(x['hard_selected'] for x in selections),
        fallback_fragments=sum(x['fallback'] for x in selections),
        fallback_reasons=dict(Counter(x['fallback_reason'] for x in selections if x['fallback'])),
        eligible_hard_objects=sorted(eligible),
        uniform_only_objects=sorted(set(ds.groups) - set(eligible)),
        visibility_cache_sha256=cache['output_sha256'], windows_sha256=analysis['windows_sha256'],
        split_hash=cache['split_hash'], mesh_hash=cache['mesh_hash'],
        source_sha256=sha(__file__),
        scope='Train-only offline sampling. Same object and noise-seed schedule. Different RGB-D fragments mean synthetic masks are not pixel-matched. No val/test examples or outcomes used for selection.',
        hard_definition='At least nine consecutive supervised natural visibility <0.3 frames whose CAD box is fully inside the image. Within each object sample physical sequence, then camera, then start uniformly.',
        rationale='Complete recovery windows are too concentrated to make a large standalone bucket. Use the broader long severe-occlusion pool, with uniform fallback for objects supported by fewer than three physical sequences.',
        provenance_limit=cache['provenance_limit'], arms={})
    for name, records in [('control', control), ('natural', curriculum)]:
        path = args.out / (name + '.json')
        path.write_text(json.dumps(records))
        descriptors = [lookup[(r['stream'], r['start'])] for r in records]
        repeats = Counter((r['stream'], r['start']) for r in records)
        report['arms'][name] = dict(summary(descriptors), manifest_sha256=sha(path),
            unique_streams=len({r['stream'] for r in records}), unique_windows=len(repeats),
            maximum_window_repeats=max(repeats.values()),
            objects=dict(sorted(Counter(r['object_id'] for r in descriptors).items())),
            physical_sequences=len({r['physical_sequence'] for r in descriptors}))
    report['uniform_matches_all_dataset_draws'] = True
    report['object_and_noise_schedules_identical'] = True
    (args.out / 'selections.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in selections))
    (args.out / 'receipt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
