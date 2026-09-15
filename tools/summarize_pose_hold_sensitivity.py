"""Reaggregate the already saved hold diagnostic using both fixed GT-prefix bins."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__);p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True);a = p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    raw = a.frames.read_bytes();rows = list(map(json.loads, raw.splitlines()));results = {}
    for field in ('initial_stationary_prefix', 'initial_stationary_prefix_loose'):
        results[field] = {}
        for subset in ('before', 'after'):
            selected = [r for r in rows if r['visibility'] < .3 and r[field] == (subset == 'before')]
            groups = defaultdict(list)
            for row in selected: groups[row['object_id']].append(row)
            results[field][subset] = dict(frames=len(selected), objects=len(groups), object_macro_percent={
                key: 100*sum(sum(r[key] for r in g)/len(g) for g in groups.values())/len(groups)
                for key in ('lip_success', 'hold_initial_success')})
    receipt = dict(completed=True, frames_sha256=hashlib.sha256(raw).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=results,
        scope='Same previously saved controlled-val occlusion frames and errors; no new inference. GT-prefix bins are posthoc, not an available deployment gate. Object sets differ between bins.')
    a.out.write_text(json.dumps(receipt, indent=2));print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
