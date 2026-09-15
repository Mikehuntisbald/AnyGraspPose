"""Export full-val figures and first-eight recovery curves after scoring."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    report=json.loads((a.run/'comparison/comparison.json').read_text());assert report['completed']
    names=list(report['checkpoints']);rows={}
    for name in names:
        folder=a.run/name/'scored';manifest=json.loads((folder/'manifest.json').read_text())
        data=(folder/'predictions.jsonl').read_bytes();assert hashlib.sha256(data).hexdigest()==manifest['predictions_sha256']
        assert manifest['checkpoint_sha256']==report['checkpoints'][name]
        rows[name]=list(map(json.loads,data.splitlines()))
    a.out.mkdir(parents=True,exist_ok=False);plt.rcParams.update({'font.size':10})
    initial=[r for r in rows['residual'] if r['initialization']]
    diagnostics=dict(initialized_streams=len(initial),bad_initializers=sum(not r['adds_005'] for r in initial),
        very_bad_initializers=sum(not r['adds_01'] for r in initial),
        initial_rotation_median_deg=float(np.median([r['rotation_deg'] for r in initial])),
        initial_center_median_mm=float(np.median([r['center_mm'] for r in initial])),models={},
        scope='Post-scoring diagnostic only; not the promotion criterion. Separates missing detections from tracking on initialized frames. Conditional populations retain the same frames for all models.')
    for name in names:
        populations=dict(all=rows[name],visibility_lt_03=[r for r in rows[name] if r['visibility'] is not None and r['visibility']<.3],
            after_initialization=[r for r in rows[name] if r['updates_after_initialization'] is not None and r['updates_after_initialization']>0],
            visibility_lt_03_after_initialization=[r for r in rows[name] if r['visibility'] is not None and r['visibility']<.3 and r['updates_after_initialization'] is not None and r['updates_after_initialization']>0])
        diagnostics['models'][name]=dict(status_counts=dict(Counter(r['status'] for r in rows[name])),
            held_updates=sum(r['pose_centered'] is not None and not r['initialization'] and r['status']!='ok' for r in rows[name]))
        for pop,rs in populations.items():
            by_object=defaultdict(list)
            for row in rs:by_object[row['object_id']].append(row)
            diagnostics['models'][name][pop]=dict(frames=len(rs),objects=len(by_object),missing=sum(r['pose_centered'] is None for r in rs),
                **{metric:100*float(np.mean([np.mean([r[metric] for r in group]) for group in by_object.values()])) for metric in ('add_01','adds_005')})
    (a.out/'initialization_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    populations=['all','visibility_lt_05','visibility_lt_03'];labels=['All','Visibility < 0.5','Visibility < 0.3']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    colors=['#606b85','#f3ab49','#7ca09b','#6978b8','#c55b69']
    for ax,pop,label in zip(axes,populations,labels):
        metric=report['populations'][pop]['metrics']['adds_005'];values=[metric['values'][n] for n in names]
        ax.bar(np.arange(len(names)),values,color=colors[:len(names)])
        for i,value in enumerate(values):ax.text(i,value+1,f'{value:.2f}',ha='center',fontsize=9)
        ax.set_xticks(np.arange(len(names)),[n.replace('_','\n') for n in names],fontsize=8)
        ax.set_title(label+f"\n{report['populations'][pop]['frames']:,} frames")
        ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Object-macro ADD-S < 0.05 diameter (%)')
    fig.suptitle('Real PoseCNN initialization, full causal LIP, zero FP — native s0 val')
    fig.tight_layout();fig.savefig(a.out/'strict_accuracy.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5));contrasts=['smooth_rotation_vs_control','smooth_rotation_vs_residual','control_vs_residual']
    for ax,pop in zip(axes,['all','visibility_lt_03']):
        stats=report['populations'][pop]['metrics']['adds_005']['comparisons']
        for i,c in enumerate(contrasts):
            d=stats[c];lo,hi=d['ci95'];ax.plot([lo,hi],[i,i],color=colors[i+2],lw=3);ax.plot(d['delta'],i,'o',color=colors[i+2])
            ax.text(.98,i+.13,f"{d['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]",transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=8)
        ax.axvline(0,color='black',lw=1,ls='--');ax.set_yticks(range(3),[c.replace('_vs_',' vs ').replace('_',' ') for c in contrasts],fontsize=9)
        ax.set_title('All' if pop=='all' else 'Visibility < 0.3');ax.set_xlabel('Strict ADD-S difference (percentage points)');ax.margins(x=.15,y=.3)
    fig.suptitle('Paired physical-sequence 95% intervals (2,000 shared draws)');fig.tight_layout();fig.savefig(a.out/'paired_strict_differences.png',dpi=160);plt.close(fig)
    # Hold the curve cohort fixed across update0..8; a last-frame detection
    # must not alter the initial point and then disappear from the curve.
    curve_streams={r['stream_id'] for r in rows['residual'] if r['initial_bad'] and r['updates_after_initialization']==8}
    curve=[];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,color in zip(names,colors):
        by_step=defaultdict(list)
        for row in rows[name]:
            k=row['updates_after_initialization']
            if row['stream_id'] in curve_streams and row['initial_bad'] and k is not None and 0<=k<=8:by_step[k].append(row)
        points={}
        for step,rs in sorted(by_step.items()):
            by_object=defaultdict(list)
            for r in rs:by_object[r['object_id']].append(r)
            entry=dict(model=name,update=step,frames=len(rs),objects=len(by_object))
            for metric in ('rotation_deg','center_mm','adds_005'):
                entry[metric]=float(np.mean([np.mean([r[metric] for r in group]) for group in by_object.values()]))*(100 if metric=='adds_005' else 1)
            curve.append(entry);points[step]=entry
        for ax,metric in zip(axes,['rotation_deg','center_mm','adds_005']):
            ax.plot(list(points),[v[metric] for v in points.values()],label=name,color=color,marker='.',lw=1.5)
    for ax,y in zip(axes,['Rotation error (degrees)','Center error (mm)','Strict ADD-S (%)']):
        ax.set_xlabel('Updates after real initializer');ax.set_ylabel(y);ax.grid(alpha=.2);ax.set_xticks(range(9))
    axes[2].legend(fontsize=8);fig.suptitle(f'Bad real initializers: fixed {len(curve_streams)}-stream cohort with eight available updates');fig.tight_layout();fig.savefig(a.out/'bad_start_first8.png',dpi=160);plt.close(fig)
    with (a.out/'bad_start_first8.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
    per_object=[]
    for name in names:
        groups=defaultdict(list)
        for row in rows[name]:groups[row['object_id']].append(row)
        for oid,group in sorted(groups.items()):
            per_object.append(dict(model=name,object_id=oid,frames=len(group),missing=sum(r['pose_centered'] is None for r in group),
                **{metric:100*float(np.mean([r[metric] for r in group])) for metric in ('add_01','adds_005')}))
    with (a.out/'per_object.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(per_object[0]));w.writeheader();w.writerows(per_object)
    lines=['# Full real-initialized s0 val screening','',report['scope'],'',
        f"Selected: **{report['selection']['selected']}**. New candidate promoted: {report['selection']['new_candidate_promoted']}.",
        '',report['selection']['limitation'],'',
        'Only control versus smooth_rotation is a matched training-budget structural comparison. The historical baselines have different cumulative training budgets. New arms are fixed final1000, seed42; no new official test is launched.',
        '', '## Object-macro scores', '', '| Population / metric | Frames | '+' | '.join(names)+' |','|---|---:|'+'---:|'*len(names)]
    for pop,metric in [('all','add_01'),('all','adds_005'),('visibility_lt_05','adds_005'),('visibility_lt_03','adds_005'),
                       ('bad_initial_first8','adds_005'),('bad_initial_first8','rotation_deg'),('bad_initial_first8','center_mm')]:
        stat=report['populations'][pop]['metrics'][metric]
        lines.append(f"| {pop} / {metric} | {stat['frames']} | "+' | '.join(f"{stat['values'][n]:.4f}" for n in names)+' |')
    lines+=['','Threshold scores are percentages; rotation is degrees and center is mm. Missing initializers remain failures in the primary score. Native-val visibility is not official BOP visib_fract, and these are not BOP AR scores.',
        '', '## Shared paired intervals', '', '| Population / metric | Contrast | Delta | 95% interval |','|---|---|---:|---:|']
    for pop,metric in [('all','adds_005'),('visibility_lt_03','adds_005'),('bad_initial_first8','rotation_deg'),('bad_initial_first8','center_mm')]:
        for contrast,stat in report['populations'][pop]['metrics'][metric]['comparisons'].items():
            lo,hi=stat['ci95'];lines.append(f"| {pop} / {metric} | {contrast} | {stat['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines+=['','2,000 shared draws over 40 physical sequences, seed20260915; not 320 independent cameras. Intervals are unadjusted for multiplicity. Point guards do not establish statistical noninferiority.',
        '', '## Detection and recovery decomposition', '',
        f"Initialized streams: {diagnostics['initialized_streams']}/320. Bad strict initializers: {diagnostics['bad_initializers']}; initializers failing ADD-S@0.1d: {diagnostics['very_bad_initializers']}. Median initial rotation/center error: {diagnostics['initial_rotation_median_deg']:.3f} degrees / {diagnostics['initial_center_median_mm']:.3f} mm.",
        '', '| Conditional population | Frames | '+' | '.join(names)+' |','|---|---:|'+'---:|'*len(names)]
    for pop in ('after_initialization','visibility_lt_03_after_initialization'):
        d=diagnostics['models']['residual'][pop];lines.append(f"| {pop} / strict ADD-S (%) | {d['frames']} | "+' | '.join(f"{diagnostics['models'][n][pop]['adds_005']:.4f}" for n in names)+' |')
    severe=diagnostics['models']['residual']['visibility_lt_03']
    lines+=['',f"Of {severe['frames']} severe-occlusion frames, {severe['missing']} have no initializer yet. These misses are shared by all models. Conditional tracking scores above are diagnostic, not an alternative promotion criterion.",
        '', '## Read activity and selection', '', json.dumps(report['smooth_activity']), '']
    for name,checks in report['selection']['checks'].items():
        failed=[k for k,v in checks.items() if not v];lines.append(f"- {name}: failed checks = {', '.join(failed) if failed else 'none'}.")
    lines+=['','## Checkpoint identities','',*['- '+n+': `'+report['checkpoints'][n]+'`.' for n in names],
        '', '## Figures', '', '![Strict accuracy](strict_accuracy.png)', '', '![Paired intervals](paired_strict_differences.png)', '', '![Bad-start recovery](bad_start_first8.png)',
        '', 'The recovery curves are descriptive point estimates. Machine-readable data: `per_object.csv`, `bad_start_first8.csv`, `initialization_diagnostics.json`; all paired frames and selection checks are in the sibling evaluation archive.']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')
    (a.out/'receipt.json').write_text(json.dumps(dict(completed=True,comparison_sha256=hashlib.sha256((a.run/'comparison/comparison.json').read_bytes()).hexdigest(),
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir()},
        scope='Post-scoring visualization. First-eight curves are descriptive point estimates, not confidence intervals. Native visibility is not official BOP visib_fract.'),indent=2))


if __name__=='__main__':main()
