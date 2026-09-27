"""Report all frozen feedback interventions; no confidence-based target filtering."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    rows = []
    for rank in range(8):
        folder = root / f'rank{rank}'
        receipt = json.loads((folder / 'receipt.json').read_text())
        assert receipt['completed'] and receipt['normal_forward_bitwise_verified']
        part = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
        assert len(part) == receipt['records'] == 8
        rows.extend(part)
    assert len({row['seed'] for row in rows}) == 64
    lines = ['# V66 geometry-to-flow feedback diagnostic', '',
             'Frozen V60-extra checkpoint; 64 training-partition physical-holdout frames.',
             'GT substitutions are diagnostics only, not deployable predictions. Original 14px pooling and feedback strength retained.',
             'Heavy means requested augmentation. Frame means; unchanged target masks, no confidence filtering.', '',
             '| Angle | Heavy | Region | Variant | Frames | Round 1 EPE px | Round 2 EPE px |',
             '|---:|---|---|---|---:|---:|---:|']
    results = {}
    for angle in (10, 60):
        for heavy in (False, True):
            for region in ('observed', 'real', 'proxy'):
                for name in rows[0]['variants']:
                    values = [r['variants'][name]['flow'] for r in rows
                              if r['angle'] == angle and r['heavy'] == heavy
                              and region in r['variants'][name]['flow']['1']]
                    if not values:
                        continue
                    epe = [statistics.mean(v[str(stage)][region]['epe'] for v in values) for stage in range(2)]
                    results[f'{angle}/{heavy}/{region}/{name}'] = dict(frames=len(values), round_epe=epe)
                    lines.append(f'| {angle} | {heavy} | {region} | {name} | {len(values)} | {epe[0]:.4f} | {epe[1]:.4f} |')
    for r in rows:
        for name, variant in r['variants'].items():
            assert variant['flow']['0'] == r['variants']['normal']['flow']['0'], (r['seed'], name)
            assert len(variant['hooks']) == 1
    results['first_round_invariance_verified'] = True
    (root / 'outcome.json').write_text(json.dumps(results, indent=2) + '\n')
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
