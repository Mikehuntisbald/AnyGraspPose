import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    logs=[[json.loads(x) for x in (a.root/'runs/seed42'/f'rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
    summary=json.loads((a.root/'outcome.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,kind in zip(axes[:2],('cad_xyz','depth')):
        for name,color in [('real','#007f86'),('proxy','#d88421')]:
            values=np.mean([[row['metrics'][f'{name}_{kind}_mm'] for row in rank] for rank in logs],axis=0)
            ax.plot(np.arange(1,len(values)+1),values,label=name,color=color)
        ax.set_title('Fixed TRAINING: '+('CAD identity XYZ' if kind=='cad_xyz' else 'depth'));ax.set_xlabel('Update');ax.set_ylabel('mm');ax.legend();ax.grid(alpha=.2)
    labels=['Real XYZ','Real depth','Proxy XYZ','Proxy depth'];keys=['real/canonical_xyz_mm','real/depth_mm','proxy/canonical_xyz_mm','proxy/depth_mm']
    x=np.arange(4)
    for offset,step,color in [(-.18,'0','#7f8792'),(.18,'200','#bf5349')]:
        axes[2].bar(x+offset,[summary['independent_heavy'][step][k] for k in keys],.36,label='step'+step,color=color)
    axes[2].set_xticks(x,labels,rotation=20);axes[2].set_ylabel('mm');axes[2].set_title('INDEPENDENT heavy probe: worse');axes[2].legend()
    fig.suptitle('Fitting works; independent accuracy does not improve. Different populations/mask policies; no pose metrics.',fontsize=10)
    fig.tight_layout();fig.savefig(a.root/'fit_vs_generalization.png',dpi=160)


if __name__=='__main__':main()
