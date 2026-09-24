"""Compare the bounded V15 trial under its immutable feature-target encoder."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root
    summaries = {step: json.loads((root/f'diagnostics/step{step}/summary.json').read_text())
                 for step in (11000, 11500, 12000)}
    source = summaries[11000]
    for value in summaries.values():
        assert value['completed'] and value['physical_sequences'] == 40
        assert value['fixed_feature_teacher'] == source['fixed_feature_teacher']
        assert value['dino_layers'] == dict(student=[4, 11], teacher=[4, 11])
        assert value['history_branch_disabled']
    selections = [
        ('spatial_hidden_real_mid', 'retrieval_top1', 'L4 patch retrieval', 100., '%'),
        ('spatial_hidden_real', 'retrieval_top1', 'L11 patch retrieval', 100., '%'),
        ('geometry_focus_real', 'xyz_mm', 'Real XYZ', 1., 'mm'),
        ('geometry_focus_real', 'depth_mm', 'Real depth', 1., 'mm'),
        ('geometry_focus_proxy', 'xyz_mm', 'CAD proxy XYZ', 1., 'mm'),
        ('geometry_focus_proxy', 'depth_mm', 'CAD proxy depth', 1., 'mm'),
        ('cad_match_real', 'top1_supported', 'CAD correspondence real', 100., '%'),
        ('cad_match_proxy', 'top1_supported', 'CAD correspondence proxy', 100., '%'),
    ]
    for source_name in ('real', 'proxy'):
        for layer in ('mid', 'last'):
            for distance in (1, 2, 4):
                for metric in ('relative_rmse', 'amplitude_ratio'):
                    selections.append((f'local_{source_name}_{layer}', f'diff{distance}_{metric}',
                                       f'{source_name} {layer} d{distance} {metric}', 1., 'ratio'))
    rows = []
    for region, metric, label, factor, unit in selections:
        values = [summaries[step]['tables']['heavy_pooled'][region] for step in summaries]
        if not all(v['sequences'] for v in values):
            continue
        assert len({v['sequences'] for v in values}) == 1
        row = dict(region=region, metric=metric, label=label, unit=unit, sequences=values[0]['sequences'])
        row.update({str(step): value['arms']['off'][metric]['mean']*factor
                    for step, value in zip(summaries, values)})
        row['final_minus_initial'] = row['12000']-row['11000']; rows.append(row)
    directory = root/'comparison'; directory.mkdir(exist_ok=True)
    with (directory/'metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, row in zip(axes.flat, rows[:4]+rows[6:8]):
        ax.plot(list(summaries), [row[str(s)] for s in summaries], marker='o')
        ax.set_title(row['label']); ax.set_ylabel(row['unit']); ax.set_xlabel('step'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(directory/'recovery.png', dpi=150); fig.savefig(directory/'recovery.pdf'); plt.close(fig)
    lines = ['# V15: completed1000-update JEPA-only trial', '',
             'Fixed40 sequences; heavy cases averaged per sequence; source11000 EMA feature targets held fixed. '
             'History disabled, pose path frozen. These are recovery metrics, not native pose results.', '',
             '| Metric |11000|11500|12000|final-initial|', '|---|---:|---:|---:|---:|']
    lines += [f"|{r['label']} ({r['unit']})|{r['11000']:.4f}|{r['11500']:.4f}|{r['12000']:.4f}|{r['final_minus_initial']:+.4f}|" for r in rows]
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (directory/'receipt.json').write_text(json.dumps(dict(completed=True, updates=1000,
        fixed_feature_teacher=source['fixed_feature_teacher'], endpoint_checkpoint=summaries[12000]['checkpoint_sha256']), indent=2))


if __name__ == '__main__':
    main()
