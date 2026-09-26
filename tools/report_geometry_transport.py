"""Paired geometry-only pilot report; never launches additional training."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); a = p.parse_args(); root = Path(a.root)
    arms = {}; rows = {}
    for arm, step in [('control',0),('control',100),('transport',0),('transport',100)]:
        name = f'{arm}{step}'; source = root/arm/'probe'/f'step{step}'
        data = []
        for rank in range(8):
            receipt = json.loads((source/f'rank{rank}'/'receipt.json').read_text())
            assert receipt['completed'] and receipt['physical_holdout'] and not receipt['teacher_inputs']
            data += [json.loads(line) for line in (source/f'rank{rank}'/'frames.jsonl').read_text().splitlines()]
        rows[name] = {d['seed']:d for d in data}
        arms[name] = {}
        for group in ('all','heavy','natural'):
            arms[name][group] = {}
            subset = [d for d in data if group=='all' or d[group]]
            for kind in ('real','proxy'):
                metrics = {}; count = 0
                for d in subset:
                    value = d['metrics'][kind]
                    if value is None: continue
                    count += 1; physical = '/'.join(d['stream'].split('|')[0].split('/')[:2])
                    for key,x in value.items():
                        if isinstance(x,(int,float)):
                            metrics.setdefault(key,{}).setdefault(physical,[]).append(x)
                arms[name][group][kind] = {key:statistics.mean(statistics.mean(x) for x in values.values()) for key,values in metrics.items()}
                arms[name][group][kind]['eligible_cases'] = count
    reference = rows['control0']
    for arm,data in rows.items():
        assert data.keys()==reference.keys()
        for seed,row in data.items():
            original = reference[seed]
            assert all(row[k]==original[k] for k in ('seed','stream','heavy','natural','window'))
            for kind in ('real','proxy'):
                x,y = row['metrics'][kind],original['metrics'][kind]
                assert (x is None)==(y is None)
                if x: assert x['pixels']==y['pixels']
    ratios = {}
    for against in ('control0','control100'):
        ratios[against] = {}
        for kind in ('real','proxy'):
            candidate = arms['transport100']['heavy'][kind]; base = arms[against]['heavy'][kind]
            ratios[against][kind] = {key:candidate[key]/base[key] for key in ('xyz_mm','depth_mm','consistency_mm')}
    passed = all(v['xyz_mm'] <= .95 and v['depth_mm'] <= 1.05 and v['consistency_mm'] <= 1.05 for reference in ratios.values() for v in reference.values())
    report = dict(completed=True, paired_inputs_verified=True, training_physical_holdout_only=True, source_step=2200,
                  updates_per_arm=100, reduction='Mean within physical sequence, then equal physical-sequence mass; empty regions excluded',
                  arms=arms, candidate_to_reference_ratios=ratios, transport_pilot_gate_passed=passed,
                  geometry_goal_complete=False, default_model_changed=False, automatic_continuation=False)
    (root/'outcome.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# V34 geometry-only 100-update pilot','','Training-partition physical holdout, same frames and targets; no official test or model promotion.','',
           '|Model|Heavy real XYZ mm|Depth mm|Proxy XYZ mm|Depth mm|','|---|---:|---:|---:|---:|']
    for name, result in arms.items():
        real,proxy=result['heavy']['real'],result['heavy']['proxy']
        lines.append(f"|{name}|{real['xyz_mm']:.3f}|{real['depth_mm']:.3f}|{proxy['xyz_mm']:.3f}|{proxy['depth_mm']:.3f}|")
    lines += ['',f'Transport pilot gate passed: {passed}. The accuracy goal remains unmet.','',
              'Gate: at least 5% lower heavy XYZ for both real and CAD-proxy targets versus source AND same-budget control; depth/consistency regression no more than 5%. No automatic extension.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
