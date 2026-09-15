"""Posthoc all/grasped visibility AR and paired physical-sequence bootstrap."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import os
import numpy as np
from streaming_bop_utils import METHODS


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--toolkit', type=Path, required=True)
    p.add_argument('--methods', nargs='+')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    protocol = json.loads((a.run/'protocol.json').read_text())
    data = Path(protocol['data_root'])
    sys.path.insert(0, str(a.toolkit))
    os.environ['DEX_YCB_DIR'] = str(data)
    from dex_ycb_toolkit.bop_eval import BOPEvaluator
    evaluator = BOPEvaluator('s0_test')
    reference = None
    per_cluster = {}
    output = dict(completed=False, visibility_definition='BOP scene_gt_info.visib_fract, not hand-only visibility',
                  unit='AR percent', bootstrap='1000 paired resamples of physical sequences (scene_id // 8), seed 20260913; descriptive posthoc intervals', methods={})
    methods = a.methods or protocol.get('methods', METHODS)
    for method in methods:
        files = sorted((a.run/method).glob('bop-*/error=*/matches_*.json'))
        assert len(files) == 120
        numerators = defaultdict(lambda: defaultdict(list))
        for f in files:
            rows = [r for r in json.loads(f.read_text()) if r['valid']]
            keys = [(r['scene_id'], r['im_id'], r['gt_id'], r['obj_id']) for r in rows]
            if reference is None:
                reference = keys
                assert len(keys) == 88014
                info = {s: json.loads((data/f'bop/s0/test/{s:06d}/scene_gt_info.json').read_text()) for s in sorted({x[0] for x in keys})}
                visibility = np.array([info[s][str(i)][g]['visib_fract'] for s,i,g,o in keys])
                grasped = np.array([evaluator._grasp_id[s][i] == o for s,i,g,o in keys])
                cluster = np.array([s//8 for s,i,g,o in keys])
                clusters, cluster_index = np.unique(cluster, return_inverse=True)
                masks = {}
                for scope, scope_mask in [('all', np.ones(len(keys), dtype=bool)), ('grasped', grasped)]:
                    for name, mask in [('all', np.ones(len(keys), dtype=bool)), ('v_lt_0.1', visibility<.1),
                                       ('v_lt_0.3', visibility<.3), ('v_lt_0.5', visibility<.5), ('v_ge_0.5', visibility>=.5)]:
                        masks[scope+'/'+name] = mask & scope_mask
                counts = {b: np.bincount(cluster_index[m], minlength=len(clusters)) for b,m in masks.items()}
                output['counts'] = {b:int(n.sum()) for b,n in counts.items()}
                output['physical_sequences'] = clusters.tolist()
            assert keys == reference, str(f)
            correct = np.array([r['est_id'] != -1 for r in rows])
            metric = f.parent.name.split('_')[0].split('=')[1]
            for bucket, mask in masks.items():
                numerators[bucket][metric].append(np.bincount(cluster_index[mask & correct], minlength=len(clusters)))
        scores = {}
        per_cluster[method] = {}
        for bucket, metrics in numerators.items():
            assert {k:len(v) for k,v in metrics.items()} == dict(vsd=100, mssd=10, mspd=10)
            averaged = {k:np.mean(v, axis=0) for k,v in metrics.items()}
            n = counts[bucket].sum()
            scores[bucket] = {k:float(v.sum()/n*100) if n else None for k,v in averaged.items()}
            success = np.mean(list(averaged.values()), axis=0)
            per_cluster[method][bucket] = success
            scores[bucket]['AR'] = float(success.sum()/n*100) if n else None
        official = json.loads((a.run/method/'results.json').read_text())
        assert abs(scores['all/all']['AR']-official['all']['mean']) < 1e-8
        assert abs(scores['grasped/all']['AR']-official['grasp_only']['mean']) < 1e-8
        output['methods'][method] = scores
        print(method, scores['all/all'], scores['all/v_lt_0.5'], flush=True)
    draws = np.random.default_rng(20260913).integers(0, len(clusters), (1000, len(clusters)))
    output['paired_differences'] = {}
    for first, second in [('lip_temporal', 'lip_no_feature_history'), ('lip_fp_temporal', 'fp_tracking')]:
        if first not in per_cluster or second not in per_cluster:
            continue
        label = first+' minus '+second
        output['paired_differences'][label] = {}
        for bucket, n in counts.items():
            if not n.sum():
                output['paired_differences'][label][bucket] = None
                continue
            delta = per_cluster[first][bucket]-per_cluster[second][bucket]
            denominator = n[draws].sum(1)
            valid = denominator > 0
            sampled = delta[draws].sum(1)[valid]/denominator[valid]*100
            output['paired_differences'][label][bucket] = dict(delta_pp=float(delta.sum()/n.sum()*100),
                ci95_pp=np.quantile(sampled, [.025,.975]).tolist(), valid_resamples=int(valid.sum()))
    output['completed'] = True
    output['per_cluster_successes'] = {m:{b:v.tolist() for b,v in values.items()} for m,values in per_cluster.items()}
    output['per_cluster_counts'] = {b:v.tolist() for b,v in counts.items()}
    (a.output or a.run/'occlusion_ar.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
