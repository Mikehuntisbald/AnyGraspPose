"""Plot completed R x K estimates and paired contrasts without recomputing scores."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    raw = args.comparison.read_bytes()
    report = json.loads(raw)
    assert report['completed'] and report['streams'] == 320
    assert report['frames_including_initialization'] == 23200
    assert report['tracked_frames'] == 22880
    assert report['initialization_uses_gt_pose'] is True
    assert report['bootstrap']['draws'] == 2000
    args.out.mkdir(parents=True, exist_ok=False)
    columns = [
        ('all', 'add_01', 'All tracking frames\nADD @ 0.1d'),
        ('visibility_lt_05', 'add_01', 'Visibility < 0.5\nADD @ 0.1d'),
        ('visibility_lt_03', 'adds_005', 'Visibility < 0.3\nADD-S @ 0.05d'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.3))
    plotted = []
    for column, (population, metric, title) in enumerate(columns):
        result = report['populations'][population]['metrics'][metric]
        ax = axes[0, column]
        for k, color in ((0, '#4776a6'), (1, '#cb6b3c')):
            values = [result['arms'][f'R{r}K{k}'] for r in (0, 1)]
            ax.plot([0, 1], values, 'o-', color=color, label=f'K{k}')
            for x, value in enumerate(values):
                other = result['arms'][f'R{x}K{1-k}']
                dy = 9 if value > other or (value == other and k) else -15
                ax.annotate(f'{value:.3f}', (x, value), xytext=(0, dy),
                            textcoords='offset points', ha='center', fontsize=9, color=color)
        values = list(result['arms'].values())
        margin = max(.08, (max(values) - min(values)) * .3)
        ax.set_ylim(min(values) - margin, max(values) + margin)
        ax.set_xlim(-.2, 1.2)
        ax.set_xticks([0, 1], ['R0', 'R1'])
        ax.set_title(title)
        ax.set_ylabel('Object-macro success (%)')
        ax.legend(frameon=False, loc='best')
        ax.grid(axis='y', alpha=.2)
        ax = axes[1, column]
        contrasts = [('R_average', 'R main effect'), ('K_average', 'K main effect'),
                     ('interaction', 'R x K interaction')]
        for y, (key, label) in enumerate(contrasts):
            contrast = result['contrasts'][key]
            lo, hi = contrast['ci95']
            ax.hlines(y, lo, hi, color='#355973', linewidth=2)
            ax.plot(contrast['delta'], y, 'o', color='#355973')
            plotted.append(dict(population=population, metric=metric,
                                contrast=key, **contrast))
        ax.axvline(0, color='black', linewidth=.8, linestyle='--')
        ax.set_yticks(range(3), [label for _, label in contrasts])
        ax.invert_yaxis()
        ax.set_ylim(2.5, -.5)
        ax.set_xlabel('Difference (percentage points)\nPaired 95% interval', fontsize=10)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Observation reliability x keyframe memory: fixed step 1,000', fontsize=14)
    fig.text(.5, .02,
             's0 val: 320 streams / 22,880 tracking frames; fixed noisy-GT first pose; zero FP.\n'
             'One training seed. Intervals resample 40 physical sequences, cameras together; not BOP AR.\n'
             'Top panels show point estimates on separate zoomed axes. Bottom panels use 2,000 shared paired draws.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .95))
    files = {}
    for extension in ('png', 'pdf', 'svg'):
        path = args.out / ('rk_factorial.' + extension)
        fig.savefig(path, dpi=180)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    receipt = dict(completed=True, comparison_sha256=hashlib.sha256(raw).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   checkpoints=report['checkpoints'], prediction_sha256=report['prediction_sha256'],
                   initial_poses_sha256=report['initial_poses_sha256'], bootstrap=report['bootstrap'],
                   scope=report['scope'], plotted_contrasts=plotted, files=files,
                   visual_review_completed=False, human_verified=False)
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(completed=True, files=files), indent=2))


if __name__ == '__main__':
    main()
