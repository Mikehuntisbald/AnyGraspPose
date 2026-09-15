"""Export matched FP/LIP figures, fixed-cohort startup curves and per-object data."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def number(value):
    # Missing-initializer success flags are explicit booleans in scored rows.
    return 1. if value=='True' else 0. if value=='False' else float(value)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=json.loads((a.comparison/'comparison.json').read_text());assert result['completed'];names=list(result['checkpoints'])
    with (a.comparison/'paired_frames.csv').open() as f:rows=list(csv.DictReader(f))
    a.out.mkdir(parents=True,exist_ok=False);plt.rcParams.update({'font.size':10})
    colors=dict(fp='#777777',residual='#4477AA',control='#EEAA33',real_mix='#228833',alignment='#993399',two_pass='#BB5566')
    populations=['all','visibility_lt_05','visibility_lt_03','bad_initial_first8']
    labels=['All frames','Visibility < 0.5','Visibility < 0.3','Bad initial pose: first 8']
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharey='row')
    for i,(pop,label) in enumerate(zip(populations,labels)):
        for j,metric in enumerate(('add_01','adds_005')):
            ax=axes[j,i];values=[result['populations'][pop]['metrics'][metric]['values'][n] for n in names]
            ax.bar(range(len(names)),values,color=[colors[n] for n in names])
            for x,value in enumerate(values):ax.text(x,value+1,f'{value:.2f}',ha='center',fontsize=9)
            ax.set_xticks(range(len(names)),[n.replace('_','\n') for n in names]);ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if j==0:ax.set_title(label+'\n'+str(result['populations'][pop]['frames'])+' frames')
    axes[0,0].set_ylabel('ADD < 0.1d (%)');axes[1,0].set_ylabel('ADD-S < 0.05d (%)')
    fig.suptitle('Same real PoseCNN initialization, full causal tracking — native s0 val')
    fig.tight_layout();fig.savefig(a.out/'accuracy.png',dpi=170);plt.close(fig)
    metrics=[('all','adds_005'),('visibility_lt_03','adds_005'),('bad_initial_first8','add_01'),('bad_initial_first8','rotation_deg')]
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    methods=[n for n in names if n!='fp']
    for ax,(pop,metric) in zip(axes.flat,metrics):
        stats=result['populations'][pop]['metrics'][metric]['comparisons']
        for i,n in enumerate(methods):
            v=stats[n+'_vs_fp'];lo,hi=v['ci95'];ax.plot([lo,hi],[i,i],color=colors[n],lw=3);ax.plot(v['delta'],i,'o',color=colors[n])
            ax.text(.98,i+.17,f"{v['delta']:+.2f} [{lo:+.2f}, {hi:+.2f}]",transform=ax.get_yaxis_transform(),ha='right',fontsize=9)
        ax.axvline(0,color='black',lw=1,ls='--');ax.set_yticks(range(len(methods)),methods);ax.set_ylim(-.6,len(methods)-.4)
        ax.set_title(pop.replace('_',' ')+' / '+metric);ax.set_xlabel('LIP minus FP (degrees; lower is better)' if metric=='rotation_deg' else 'LIP minus FP (percentage points; higher is better)');ax.margins(x=.2)
    fig.suptitle('Paired physical-sequence 95% intervals; 2,000 shared draws, unadjusted')
    fig.tight_layout();fig.savefig(a.out/'paired_intervals.png',dpi=170);plt.close(fig)
    cohort={r['stream_id'] for r in rows if r['initial_bad']=='True' and r['updates_after_initialization']=='8'}
    curve=[];fig,axes=plt.subplots(1,4,figsize=(16,4))
    for n in names:
        points=[]
        for step in range(9):
            group=[r for r in rows if r['stream_id'] in cohort and r['updates_after_initialization']==str(step)]
            assert len(group)==len(cohort)
            objects=defaultdict(list)
            for row in group:objects[row['object_id']].append(row)
            value=dict(method=n,update=step,streams=len(cohort),objects=len(objects))
            for metric in ('rotation_deg','center_mm','add_01','adds_005'):
                value[metric]=float(np.mean([np.mean([number(r[n+'_'+metric]) for r in rs]) for rs in objects.values()]))*(100 if metric in ('add_01','adds_005') else 1)
            points.append(value);curve.append(value)
        for ax,metric in zip(axes,('rotation_deg','center_mm','add_01','adds_005')):ax.plot(range(9),[v[metric] for v in points],label=n,color=colors[n],marker='.')
    for ax,y in zip(axes,('Rotation (degrees)','Center (mm)','ADD < 0.1d (%)','ADD-S < 0.05d (%)')):
        ax.set_ylabel(y);ax.set_xlabel('Updates after initializer');ax.set_xticks(range(9));ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.suptitle(f'Fixed {len(cohort)}-stream bad-start cohort; descriptive curves')
    fig.tight_layout();fig.savefig(a.out/'bad_start_first8.png',dpi=170);plt.close(fig)
    with (a.out/'bad_start_first8.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
    per_object=[]
    for population in populations:
        def selected(r):
            if population=='all':return True
            if population=='bad_initial_first8':return r['initial_bad']=='True' and r['updates_after_initialization'] and 0<int(r['updates_after_initialization'])<=8
            return r['visibility'] and float(r['visibility'])<(.5 if population=='visibility_lt_05' else .3)
        groups=defaultdict(list)
        for r in rows:
            if selected(r):groups[r['object_id']].append(r)
        for oid,group in sorted(groups.items(),key=lambda x:int(x[0])):
            for n in names:
                v=dict(population=population,object_id=oid,method=n,frames=len(group))
                for metric in ('add_01','adds_01','adds_005'):v[metric]=100*float(np.mean([number(r[n+'_'+metric]) for r in group]))
                per_object.append(v)
    with (a.out/'per_object.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(per_object[0]));w.writeheader();w.writerows(per_object)
    report=(a.comparison/'report.md').read_text()+'\n![Accuracy](accuracy.png)\n\n![Paired intervals](paired_intervals.png)\n\n![First eight updates](bad_start_first8.png)\n'
    (a.out/'report.md').write_text(report)
    (a.out/'receipt.json').write_text(json.dumps(dict(completed=True,comparison_sha256=hashlib.sha256((a.comparison/'comparison.json').read_bytes()).hexdigest(),
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir()}),indent=2))


if __name__=='__main__':main()
