"""Bind a fresh counter range to the existing official-train uniform sampler."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lip.data.stream_clips import StreamClips
from lip.engine.stream_checkpoint import source_hash


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('data-root', 'index-root', 'out'): p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--offset', type=int, required=True);p.add_argument('--count', type=int, default=64000)
    p.add_argument('--seed', type=int, default=42);a = p.parse_args()
    if a.offset < 0 or a.count < 1: p.error('Nonnegative offset and positive count required')
    ds = StreamClips(a.data_root, a.index_root, 8, 48, seed=a.seed, start_sample=a.offset)
    samples = [ds.choose(i) for i in range(a.count)];a.out.mkdir(parents=True, exist_ok=False)
    for item in samples:
        assert item['start'] in ds.starts[item['stream']]
        assert ds.streams[item['stream']]['split'] == 'train'
    path = a.out/'training_samples.json';path.write_text(json.dumps(samples))
    counts = Counter((r['stream'], r['start']) for r in samples)
    receipt = dict(completed=True, source_sha256=source_hash(), generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), entries=len(samples),
        seed=a.seed, sample_counter_range=[a.offset, a.offset+a.count-1], split_hash=ds.audit['split_hash'], mesh_hash=ds.audit['mesh_hash'],
        eligible_streams=sum(bool(len(s)) for s in ds.starts), actual_unique_streams=len({r['stream'] for r in samples}),
        unique_windows=len(counts), maximum_window_repeats=max(counts.values()),
        object_counts=dict(sorted(Counter(ds.streams[r['stream']]['object_id'] for r in samples).items())),
        scope='Exact StreamClips.choose draws from official train, no image/visibility/held-out access. Counter range advances; this does not claim previously unseen fragments.',
        training_started=False)
    (a.out/'receipt.json').write_text(json.dumps(receipt, indent=2));print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
