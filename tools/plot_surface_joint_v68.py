"""Standalone paired geometry figure; original evaluation regions remain fixed."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root
    data=json.loads((root/'outcome.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),constrained_layout=True)
    colors=['#9199a4','#4675ba','#dc8542']
    for column,angle in enumerate((0,10,60)):
        for row,metric in enumerate(('canonical_xyz_mm','depth_mm')):
            ax=axes[row,column]
            for offset,arm in enumerate(('baseline','control','surface')):
                values=[data[f'{arm}_{angle}']['heavy_'+region][metric] for region in ('real','proxy')]
                ax.bar([x+(offset-1)*.24 for x in range(2)],values,width=.23,color=colors[offset],label=arm)
            ax.set_xticks([0,1],['Artificially hidden real','CAD proxy'])
            ax.set_ylabel(('Canonical XYZ' if row==0 else 'Depth')+' error (mm)')
            ax.set_title(f'{angle}° initial rotation error')
            ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('V68r1: paired 200-update training; requested heavy occlusion\nTraining-partition physical holdout; lower is better; no pose evaluation',fontsize=12)
    fig.savefig(root/'paired_geometry.png',dpi=180)
    fig.savefig(root/'paired_geometry.pdf')
    plt.close(fig)


if __name__=='__main__':main()
