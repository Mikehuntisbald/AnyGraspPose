"""Export recovery-only paired scores and the predetermined rank0 heavy frame."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result=json.loads((a.root/'outcome.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    keys=['real/canonical_xyz_mm','real/depth_mm','proxy/canonical_xyz_mm','proxy/depth_mm']
    labels=['Real CAD XYZ','Real depth','Proxy CAD XYZ','Proxy depth'];x=np.arange(4)
    for ax,split in zip(axes,('paired','confirmation')):
        data=result[split]['heavy']
        names=['control_0','control_200','balanced_200'] if split=='paired' else ['source','control','balanced']
        for offset,name,label,color in zip((-.25,0,.25),names,('Source','CE 1.0','CE 0.1'),('#8b939c','#d18736','#258b86')):
            ax.bar(x+offset,[data[name][k] for k in keys],width=.24,label=label,color=color)
        ax.set_xticks(x,labels,rotation=15);ax.set_ylabel('mm (lower is better)');ax.legend()
        ax.set_title('Heavy: '+('usual development64' if split=='paired' else 'prespecified confirmation64'))
    fig.suptitle('Same initialization/data,200 updates; original masks. Canonical identity and depth are different targets.',fontsize=10)
    fig.tight_layout();fig.savefig(a.root/'paired_recovery.png',dpi=160);plt.close(fig)
    examples=[np.load(a.root/'confirmation'/arm/'rank0/heavy_example.npz') for arm in ('source','control','balanced')]
    first=examples[0];truth=first['target_depth'].squeeze()
    mask=first['real_mask'].squeeze()|first['proxy_mask'].squeeze()
    for e in examples[1:]:
        for key in ('target_depth','target_xyz','real_mask','proxy_mask','occluded_rgb'):
            assert np.array_equal(e[key],first[key],equal_nan=True),key
    lo,hi=np.percentile(truth[mask],[1,99]);d=float(first['diameter'])
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    for row,(name,e) in enumerate(zip(('Source','CE 1.0','CE 0.1'),examples)):
        rgb=e['occluded_rgb'].squeeze().transpose(1,2,0)
        axes[row,0].imshow(np.clip(rgb,0,1));axes[row,0].set_title(name+' / student input')
        axes[row,1].imshow(np.where(mask,truth,np.nan),vmin=lo,vmax=hi,cmap='viridis');axes[row,1].set_title('Target depth (m)')
        depth=e['predicted_depth'].squeeze()
        axes[row,2].imshow(np.where(mask,depth,np.nan),vmin=lo,vmax=hi,cmap='viridis');axes[row,2].set_title('Recovered depth (same scale)')
        im=axes[row,3].imshow(np.where(mask,np.abs(depth-truth)*1000,np.nan),vmin=0,vmax=30,cmap='magma')
        axes[row,3].set_title('Depth error (mm,0–30)')
        for ax in axes[row]:ax.set_axis_off()
    fig.suptitle('Fixed confirmation rank0, item2; unfiltered real/proxy target masks. Not selected by improvement.',fontsize=11)
    fig.tight_layout();fig.savefig(a.root/'fixed_recovery_example.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
