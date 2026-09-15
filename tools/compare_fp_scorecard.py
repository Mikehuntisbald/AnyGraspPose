"""Matched zero-FP LIP versus frozen FP, with shared populations and intervals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_rk_ablation import paired_summary
from diagnose_occlusion_events import stable_recovery, successful

SUCCESS = ('add_01', 'adds_01', 'adds_005')
ERRORS = ('add_m', 'adds_m', 'center_mm', 'rotation_deg')


def episodes(rows):
    """No episode/window crosses a gap, unknown visibility, init, or no-pose frame."""
    groups = {}
    for row in rows:
        groups.setdefault(row['stream_id'], []).append(row)
    result = []
    for sid, stream in sorted(groups.items()):
        stream.sort(key=lambda r: r['frame_index'])
        usable = lambda r: not r['initialization'] and r['updates_after_initialization'] is not None and r['visibility'] is not None
        i = 0
        while i < len(stream):
            if not usable(stream[i]) or stream[i]['visibility'] >= .5:
                i += 1
                continue
            j = i + 1
            while j < len(stream) and usable(stream[j]) and stream[j]['visibility'] < .5 and stream[j]['frame_index'] == stream[j-1]['frame_index'] + 1:
                j += 1
            after = []
            reason = 'stream_end'
            previous = stream[j-1]['frame_index']
            for row in stream[j:j+10]:
                if row['frame_index'] != previous + 1:
                    reason = 'frame_gap'; break
                if not usable(row):
                    reason = 'initialization_or_missing_pose_or_visibility'; break
                if row['visibility'] < .5:
                    reason = 'next_occlusion'; break
                after.append(row['frame_index']); previous = row['frame_index']
            result.append(dict(stream_id=sid, object_id=stream[i]['object_id'], physical_sequence=stream[i]['physical_sequence'],
                frames=[r['frame_index'] for r in stream[i:j]], post_frames=after, length=j-i,
                post_complete=len(after) == 10, censor_reason=None if len(after) == 10 else reason))
            i = j
    return result


def load_evaluations(folders):
    data, manifests = {}, {}
    for name, folder in folders.items():
        folder = Path(folder); m = json.loads((folder/'manifest.json').read_text())
        assert m['completed'] and m['population_verified'] and m['split'] == 'val'
        assert m['frames'] == 23200 and len(m['streams']) == 320 and not m['initialization_uses_gt_pose']
        if name == 'fp':
            assert m['architecture_id'] == 'foundationpose_tracking_v1' and m['config']['iterations'] == 2
            assert not m['config']['registration'] and not m['config']['initial_refinement'] and m['fp_calls'] == 22303
        else:
            assert m['fp_calls'] == 0
        raw = (folder/'predictions.jsonl').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == m['predictions_sha256']
        rows = list(map(json.loads, raw.splitlines()))
        data[name] = {(r['stream_id'], r['frame_index']): r for r in rows}
        assert len(rows) == len(data[name]) == 23200
        manifests[name] = m
    ref = data['residual']; rm = manifests['residual']
    for name, rows in data.items():
        assert rows.keys() == ref.keys()
        assert all(manifests[name][k] == rm[k] for k in ('split_hash', 'mesh_hash', 'initializers_sha256'))
        for key, row in ref.items():
            other = rows[key]
            assert all(row[k] == other[k] for k in ('object_id', 'physical_sequence', 'visibility', 'initialization', 'updates_after_initialization', 'initial_bad'))
            if row['initialization']:
                # Original PoseCNN poses are the exact shared inference input.
                assert row['pose_original'] == other['pose_original']
                np.testing.assert_allclose(row['pose_centered'], other['pose_centered'], atol=2e-7, rtol=0)
    return data, manifests


def statistic(values, objects, sequences, clusters, weights, names):
    selected = weights[:, np.isin(clusters, np.unique(sequences))]
    point, draws = paired_summary(values, objects, sequences, selected)
    comparisons = {}
    for first in names:
        for second in ('fp', 'residual', 'control'):
            if first == second or second not in names or first == 'fp':
                continue
            delta = draws[:, names.index(first)] - draws[:, names.index(second)]
            finite = delta[np.isfinite(delta)]
            comparisons[first+'_vs_'+second] = dict(delta=float(point[names.index(first)]-point[names.index(second)]),
                ci95=np.quantile(finite, [.025, .975]).tolist(), valid_draws=len(finite))
    return dict(frames=len(values), values=dict(zip(names, point.tolist())), comparisons=comparisons)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--evaluation', required=True, action='append'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rotation-alignment',action='store_true')
    a = p.parse_args(); folders = dict(x.split('=', 1) for x in a.evaluation)
    assert len(folders) == len(a.evaluation) and {'fp', 'residual'} <= folders.keys()
    data, manifests = load_evaluations(folders); names = list(data); keys = sorted(data['residual'])
    rows = [data['residual'][k] for k in keys]; objects = np.array([r['object_id'] for r in rows])
    sequences = np.array([r['physical_sequence'] for r in rows]); clusters = np.unique(sequences)
    weights = np.random.default_rng(20260915).multinomial(len(clusters), np.ones(len(clusters))/len(clusters), size=2000)
    es = episodes(rows); long_keys = {(e['stream_id'], f) for e in es if e['length'] > 8 for f in e['frames']}
    updates = np.array([r['updates_after_initialization'] is not None and r['updates_after_initialization'] > 0 for r in rows])
    bad = np.array([r['initial_bad'] is True for r in rows])
    first8 = np.array([r['updates_after_initialization'] is not None and 0 < r['updates_after_initialization'] <= 8 for r in rows])
    masks = dict(all=np.ones(len(keys), bool), updates=updates,
        visibility_lt_05=np.array([r['visibility'] is not None and r['visibility'] < .5 for r in rows]),
        visibility_lt_03=np.array([r['visibility'] is not None and r['visibility'] < .3 for r in rows]),
        visibility_ge_05=np.array([r['visibility'] is not None and r['visibility'] >= .5 for r in rows]),
        long_occlusion_gt8=np.array([k in long_keys for k in keys]), initial_bad=bad & updates,
        initial_good=(~bad) & updates, bad_initial_first8=bad & first8)
    result = dict(completed=True, scope='Native s0 val: 320 grasped-object streams, 23200 frames, common frozen first-legal PoseCNN initialization. LIP uses zero FP. Pure FP uses track_one(iteration=2). Not official test BOP AR or SOTA.',
        checkpoints={n:m['checkpoint_sha256'] for n,m in manifests.items()}, sources={n:m['source_sha256'] for n,m in manifests.items()},
        predictions={n:m['predictions_sha256'] for n,m in manifests.items()}, initializers_sha256=manifests['residual']['initializers_sha256'],
        bootstrap=dict(draws=2000, seed=20260915, physical_sequences=len(clusters), weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
            multiplicity_adjusted=False, scope='Population intervals, one training seed; not seed stability.'),
        estimator='Object-macro frame metrics. Missing initializations fail success metrics; continuous errors use common emitted-pose frames. Visibility is the native GT-derived scoring proxy, never an inference input.',
        populations={}, recovery={}, promotion={}, fp_scorecard={})
    flat, paired, cluster_rows = [], [], []
    for population, mask in masks.items():
        metrics = {}; result['populations'][population] = dict(frames=int(mask.sum()), metrics=metrics)
        for metric in SUCCESS + ERRORS:
            use = mask.copy()
            if metric in ERRORS:
                use &= np.array([all(data[n][k][metric] is not None for n in names) for k in keys])
            if not use.any():
                continue
            selected = [k for k, yes in zip(keys, use) if yes]
            scale = 100 if metric in SUCCESS else 1
            values = np.array([[data[n][k][metric]*scale for n in names] for k in selected])
            assert np.isfinite(values).all()
            stats = statistic(values, objects[use], sequences[use], clusters, weights, names); metrics[metric] = stats
            for n in names:
                flat.append(dict(population=population, metric=metric, method=n, frames=len(selected), value=stats['values'][n]))
            for seq in np.unique(sequences[use]):
                for obj in np.unique(objects[use][sequences[use] == seq]):
                    take = (sequences[use] == seq) & (objects[use] == obj)
                    for ni, n in enumerate(names):
                        cluster_rows.append(dict(population=population, metric=metric, physical_sequence=seq, object_id=int(obj), method=n,
                            frame_count=int(take.sum()), sum=float(values[take, ni].sum())))
    for key in keys:
        r = data['residual'][key]
        row = {k:r[k] for k in ('stream_id','frame_index','object_id','physical_sequence','visibility','initialization','initial_bad','updates_after_initialization')}
        for n in names:
            for metric in SUCCESS + ERRORS: row[n+'_'+metric] = data[n][key][metric]
        paired.append(row)
    for e in es:
        e['arms'] = {}
        sid = e['stream_id']
        for n in names:
            end = data[n][(sid, e['frames'][-1])]
            post = [data[n][(sid, f)] for f in e['post_frames']]
            e['arms'][n] = dict(end_success=successful(end), stable_offset=stable_recovery(post))
        # Fixed residual-defined failure population; same events for all methods.
        e['reference_failed_at_end'] = not e['arms']['residual']['end_success']
    event_flat = []
    for population, selected in [('all', es), ('long_gt8', [e for e in es if e['length'] > 8])]:
        complete = [e for e in selected if e['post_complete']]
        fixed = [e for e in complete if e['reference_failed_at_end']]
        report = dict(episodes=len(selected), complete_windows=len(complete), censored=len(selected)-len(complete),
            fixed_reference_failed_events=len(fixed), conditional_by_method={})
        for n in names:
            failed = [e for e in complete if not e['arms'][n]['end_success']]
            recovered = [e for e in failed if e['arms'][n]['stable_offset'] is not None]
            report['conditional_by_method'][n] = dict(failed_at_end=len(failed), recovered_stably=len(recovered),
                rate=len(recovered)/len(failed) if failed else None)
        if fixed:
            obj = np.array([e['object_id'] for e in fixed]); seq = np.array([e['physical_sequence'] for e in fixed])
            values = np.array([[100.*(e['arms'][n]['stable_offset'] is not None) for n in names] for e in fixed])
            report['fixed_population_stable_success'] = statistic(values, obj, seq, clusters, weights, names)
        result['recovery'][population] = report
    result['recovery']['definition'] = 'Three consecutive accurate (ADD-S<.05d), status-ok clear frames; full ten-frame clear window. Gaps/init/missing visibility censor windows. Conditional denominators vary and are descriptive only. Paired rate uses frozen residual-failed events for every method; it measures post-event stable success, not a causal recovery effect.'
    for e in es:
        for n in names:
            event_flat.append(dict(stream_id=e['stream_id'], object_id=e['object_id'], physical_sequence=e['physical_sequence'], start=e['frames'][0],
                end=e['frames'][-1], length=e['length'], post_complete=e['post_complete'], censor_reason=e['censor_reason'], reference_failed_at_end=e['reference_failed_at_end'], method=n, **e['arms'][n]))
    for n in names:
        if n == 'fp': continue
        entries = []
        for population, report in result['populations'].items():
            for metric, stats in report['metrics'].items():
                change = stats['comparisons'][n+'_vs_fp']; direction = 1 if metric in SUCCESS else -1
                entries.append(dict(population=population, metric=metric, favorable_delta=direction*change['delta'],
                    favorable_ci95=sorted([direction*x for x in change['ci95']])))
        for population in ('all', 'long_gt8'):
            stats = result['recovery'][population].get('fixed_population_stable_success')
            if stats:
                change = stats['comparisons'][n+'_vs_fp']
                entries.append(dict(population='recovery_'+population, metric='fixed_stable_success', favorable_delta=change['delta'], favorable_ci95=change['ci95']))
        result['fp_scorecard'][n] = dict(entries=entries, accuracy_entries=len(entries), favorable_points=sum(x['favorable_delta']>0 for x in entries),
            unfavorable_points=sum(x['favorable_delta']<0 for x in entries), all_accuracy_points_favorable=all(x['favorable_delta']>0 for x in entries),
            runtime_measured_here=False, all_metrics_superior=False, limitation='Unadjusted per-entry intervals; no universal dominance claim. Runtime/memory require separate isolated measurement.')
    def delta(n, pop, metric): return result['populations'][pop]['metrics'][metric]['comparisons'][n+'_vs_residual']
    for n in ('control', 'real_mix'):
        if n not in names: continue
        checks = dict(overall_add_protected=delta(n,'all','add_01')['delta']>=0, overall_strict_protected=delta(n,'all','adds_005')['delta']>=0,
            severe_strict_protected=delta(n,'visibility_lt_03','adds_005')['delta']>=0)
        checks['paired_fp_deficit_gain'] = any([delta(n,'visibility_lt_05','add_01')['ci95'][0]>0,
            delta(n,'visibility_lt_03','add_01')['ci95'][0]>0, delta(n,'bad_initial_first8','adds_005')['ci95'][0]>0,
            delta(n,'bad_initial_first8','rotation_deg')['ci95'][1]<0])
        result['promotion'][n] = dict(checks=checks, eligible=all(checks.values()))
    if a.rotation_alignment:
        assert 'alignment' in names and 'control' in names
        n='alignment'
        matched=result['populations']['bad_initial_first8']['metrics']['rotation_deg']['comparisons']['alignment_vs_control']
        checks=dict(overall_add_protected=delta(n,'all','add_01')['delta']>=0,
            overall_strict_protected=delta(n,'all','adds_005')['delta']>=0,
            severe_strict_protected=delta(n,'visibility_lt_03','adds_005')['delta']>=0,
            startup_center_protected=delta(n,'bad_initial_first8','center_mm')['delta']<=0,
            startup_rotation_parent_ci=delta(n,'bad_initial_first8','rotation_deg')['ci95'][1]<0,
            startup_rotation_control_ci=matched['ci95'][1]<0)
        result['promotion']={'alignment':dict(checks=checks,eligible=all(checks.values()))}
    eligible = [n for n,v in result['promotion'].items() if v['eligible']]
    result['selected'] = max(eligible, key=lambda n: result['populations']['visibility_lt_03']['metrics']['adds_005']['values'][n]) if eligible else 'residual'
    result['selection_limitation'] = 'Fixed final1000 only. Point protection is not statistical noninferiority; seed replication and frozen official test remain necessary. No automatic test launch.'
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)); np.save(a.out/'bootstrap_weights.npy', weights)
    for name, export in [('metrics',flat),('paired_frames',paired),('paired_physical_sequences',cluster_rows),('paired_events',event_flat)]: write_csv(a.out/(name+'.csv'),export)
    lines = ['# Matched LIP and FoundationPose scorecard', '', result['scope'], '', result['estimator'], '',
        '| Population / metric | '+' | '.join(names)+' |', '|---|'+'---:|'*len(names)]
    for population in ('all','visibility_lt_05','visibility_lt_03','bad_initial_first8'):
        for metric in SUCCESS+('rotation_deg','center_mm'):
            stats = result['populations'][population]['metrics'][metric]
            lines.append('| '+population+' / '+metric+' | '+' | '.join(f"{stats['values'][n]:.3f}" for n in names)+' |')
    lines += ['', 'Selected: '+result['selected']+'.', result['selection_limitation'], '', 'All remaining FP deficits and paired intervals are in comparison.json; method-dependent recovery denominators are reported separately.']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))


if __name__ == '__main__': main()
