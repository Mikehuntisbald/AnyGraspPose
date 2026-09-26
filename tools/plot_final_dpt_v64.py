"""V64 controlled final-only geometry comparison."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    data=json.loads((a.root/'outcome.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for ax,(region,metric) in zip(axes.flat,[(r,m) for r in ('real','proxy') for m in ('canonical_xyz_mm','depth_mm')]):
        for i,arm in enumerate(('baseline','matched_control','final_only')):
            values=[data[f'{angle}/{arm}/{region}'][metric] for angle in (0,10,60)]
            ax.bar(np.arange(3)+(i-1)*.24,values,.24,label=arm)
        ax.set_xticks(range(3),['0 deg','10 deg','60 deg']);ax.set_ylabel('mm');ax.set_title(region+' / '+metric.replace('_mm',''));ax.grid(axis='y',alpha=.2)
    axes[0,0].legend();fig.suptitle('Requested heavy occlusion; controlled physical-holdout probes')
    fig.savefig(a.root/'geometry_comparison.png',dpi=160)

if __name__=='__main__':main()
