"""Shareable full-validation comparisons and a predetermined error-map example."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.root/'outcome.json').read_text());m=r['metrics']
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for col,case in enumerate(('natural','light','heavy')):
        regions=('proxy',) if case=='natural' else ('real','proxy')
        for row,key,title in ((0,'canonical_xyz_mm','CAD correspondence XYZ'),(1,'depth_mm','Camera depth')):
            ax=axes[row,col];x=np.arange(len(regions))
            for offset,arm,color,label in ((-.18,'v52','#8b949e','V52 source'),(.18,'v53','#258b86','V53 +5000')):
                values=[m[f'{case}/{arm}/{region}/{key}']['mean'] for region in regions]
                bars=ax.bar(x+offset,values,.36,color=color,label=label)
                ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
            ax.set_xticks(x,['Artificially hidden real' if s=='real' else 'Naturally hidden CAD proxy' for s in regions],fontsize=8)
            ax.set_ylabel('mm (lower is better)');ax.set_title(case+' / '+title);ax.legend(fontsize=8);ax.margins(y=.2)
    fig.suptitle('Full s0 validation / 320 streams: shared native-reference crops, equal physical-sequence means\nErrors require legal input and nonempty targets; coverage is reported separately. No pose evaluation.',fontsize=10)
    fig.savefig(a.root/'full_recovery.png',dpi=170);plt.close(fig)
    strata=r['strata'];fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    bins=('le15','15to45','gt45')
    for row,region in enumerate(('real','proxy')):
        for col,key in enumerate(('canonical_xyz_mm','depth_mm')):
            ax=axes[row,col]
            for arm,color in (('v52','#8b949e'),('v53','#258b86')):
                values=[strata.get(f'base_rotation/{b}/heavy/{arm}/{region}/{key}',{}).get('mean',np.nan) for b in bins]
                ax.plot(np.arange(3),values,'o-',label=arm,color=color)
            ax.set_xticks(np.arange(3),['≤15°','15–45°','>45°']);ax.set_ylabel('mm');ax.set_xlabel('Input base rotation error');ax.set_title(region+' / '+key);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Heavy added occlusion, stratified by shared base error (unreduced angle)\nDifferent bin populations; descriptive stratification, not a causal rotation experiment.',fontsize=10)
    fig.savefig(a.root/'base_rotation_strata.png',dpi=170);plt.close(fig)
    path=a.root/'rank0/fixed_example.npz'
    if path.exists():
        z=np.load(path);lane=2;mask=z['real_mask'][lane,0]|z['proxy_mask'][lane,0];truth=z['target_depth'][lane,0]
        if mask.any():
            low,high=np.percentile(truth[mask],[1,99]);fig,axes=plt.subplots(2,5,figsize=(15,6),layout='constrained')
            owner=np.zeros((*mask.shape,3));owner[z['real_mask'][lane,0]]=[.95,.45,.15];owner[z['proxy_mask'][lane,0]]=[.1,.6,.9]
            for row,arm in enumerate(('v52','v53')):
                depth=z[arm+'_surface_depth_m'][lane,0]
                panels=[z['input_rgb'][lane].transpose(1,2,0),owner,np.where(mask,truth,np.nan),np.where(mask,depth,np.nan),np.where(mask,np.abs(depth-truth)*1000,np.nan)]
                titles=[arm+' input','Real orange / proxy blue',f'Target [{low:.3f},{high:.3f}] m','Recovered (same depth scale)','Depth error [0,50] mm']
                for col,(panel,title) in enumerate(zip(panels,titles)):
                    axes[row,col].imshow(panel,**(dict(vmin=low,vmax=high,cmap='viridis') if col in (2,3) else dict(vmin=0,vmax=50,cmap='magma') if col==4 else {}))
                    axes[row,col].set_title(title,fontsize=9);axes[row,col].set_axis_off()
            fig.suptitle('Predetermined rank0 first stream, initialization+24 / heavy: same inputs and targets; not selected by improvement.',fontsize=10)
            fig.savefig(a.root/'fixed_example.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
