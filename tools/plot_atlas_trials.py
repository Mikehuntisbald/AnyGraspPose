"""Figures from sealed V43/V44 paired metrics, never training-loss proxies."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    v43=a.root/'atlas_direct_v43';v44=a.root/'atlas_prior_v44'
    matching=json.loads((v43/'learned_only_outcome.json').read_text())['metrics']
    standard=json.loads((v43/'outcome.json').read_text())['metrics']
    prior=json.loads((v44/'outcome.json').read_text())['metrics']
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    data=[(axes[0],matching,['assisted','direct'],'V43: learned matching alone'),
          (axes[1],standard,['assisted_100_corrupted','direct_100_corrupted'],'V43: complete inference'),
          (axes[2],prior,['control_100_corrupted','supervised_100_corrupted'],'V44: final XYZ supervision')]
    for ax,values,keys,title in data:
        for i,key in enumerate(keys):
            label=key.split('_')[0]
            ax.bar(np.arange(2)+(i-.5)*.35,[values[key][r]['canonical_xyz_mm'] for r in ('real','proxy')],.35,label=label)
        ax.set_xticks([0,1],['Real region','CAD proxy']);ax.set_ylabel('Canonical XYZ error (mm)');ax.set_title(title);ax.legend()
    fig.suptitle('Fixed heavy probes, original masks retained; lower is better')
    fig.tight_layout();fig.savefig(v44/'matching_and_prior.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
