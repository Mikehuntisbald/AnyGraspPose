"""Posthoc visibility-subset AR from existing official BOP matches; no inference."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    base = a.runtime / 'runs/official_s0_test'
    methods = {n: base / n for n in ('posecnn_fp', 'posecnn_lip_fp', 'posecnn_lip', 'deepim_rgbd_reference')}
    methods['posecnn'] = a.runtime / 'runs/official_posecnn_final'
    info = {}
    result = {'protocol': 'Posthoc BOP visib_fract subsets, all official valid targets including missing detections; not paper hand-only visibility or paper AUC.', 'methods': {}}
    reference_keys = None
    for name, folder in methods.items():
        files = sorted(folder.glob('bop-*/error=*/matches_*.json'))
        assert len(files) == 120, (name, len(files))
        scores = defaultdict(lambda: defaultdict(list))
        for f in files:
            matches = [m for m in json.loads(f.read_text()) if m['valid']]
            keys = [(m['scene_id'], m['im_id'], m['gt_id'], m['obj_id']) for m in matches]
            if reference_keys is None:
                reference_keys = keys
                assert len(keys) == 88014
                for scene in sorted({k[0] for k in keys}):
                    info[scene] = json.loads((a.data_root / f'bop/s0/test/{scene:06d}/scene_gt_info.json').read_text())
                visibility = np.array([info[s][str(i)][g]['visib_fract'] for s,i,g,o in keys])
                assert np.isfinite(visibility).all()
                masks = {'all': np.ones(len(keys), dtype=bool), 'v_lt_0.1': visibility < .1,
                         'v_lt_0.3': visibility < .3, 'v_lt_0.5': visibility < .5,
                         'v_ge_0.5': visibility >= .5}
                result['counts'] = {k: int(v.sum()) for k,v in masks.items()}
            assert keys == reference_keys, str(f)
            correct = np.array([m['est_id'] != -1 for m in matches])
            metric = f.parent.name.split('_')[0].split('=')[1]
            for bucket, mask in masks.items():
                scores[bucket][metric].append(float(correct[mask].mean()*100) if mask.any() else None)
        summary = {}
        for bucket, metrics in scores.items():
            assert {k: len(v) for k,v in metrics.items()} == {'mspd': 10, 'mssd': 10, 'vsd': 100}
            row = {k: float(np.mean(v)) if v[0] is not None else None for k,v in metrics.items()}
            row['AR'] = float(np.mean(list(row.values()))) if row['vsd'] is not None else None
            summary[bucket] = row
        official = json.loads((folder/'results.json').read_text())
        assert abs(summary['all']['AR']-official['all']['mean']) < 1e-8
        result['methods'][name] = summary
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2)+'\n')
        print(name, summary, flush=True)
    result['completed'] = True
    result['checks'] = {'official_match_files_per_method': 120,
                        'identical_ordered_valid_target_keys': len(reference_keys),
                        'all_AR_matches_official_tolerance': 1e-8}
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
