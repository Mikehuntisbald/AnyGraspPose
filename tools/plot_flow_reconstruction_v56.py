"""Fixed first-heavy rank0 example, no selection by improvement."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root)
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for row,tag in enumerate(('initial','final_on')):
        path=root/'probe'/tag/'rank0'
        data=np.load(path/'heavy_example.npz');flow=np.load(path/'flow_example.npz')
        image=data['occluded_rgb'][0].transpose(1,2,0)
        domain=(data['real_mask']|data['proxy_mask'])[0,0]
        xyz=np.linalg.norm(data['predicted_xyz']-data['cad_target_xyz'],axis=1)[0]*float(data['diameter'])*1000
        depth=np.abs(data['predicted_depth']-data['target_depth'])[0,0]*1000
        ax=axes[row,0];ax.imshow(image)
        for name,color in [('observed','deepskyblue'),('real','orange'),('proxy','magenta')]:
            mask=flow[name][0]
            pred,truth=flow['uv1'][0,mask],flow['target_uv'][0,mask]
            ax.scatter(truth[:,0],truth[:,1],s=18,marker='x',color=color,label=name)
            for p,t in zip(pred,truth):ax.plot([p[0],t[0]],[p[1],t[1]],color=color,lw=.7)
            ax.scatter(pred[:,0],pred[:,1],s=10,facecolors='none',edgecolors=color)
        ax.set_title(f'{tag}: circle=prediction, x=GT endpoint');ax.legend(fontsize=7,loc='lower right')
        for col,error,title in [(1,xyz,'Canonical surface identity error'),(2,depth,'Depth error')]:
            ax=axes[row,col];ax.imshow(image)
            shown=ax.imshow(np.ma.array(error,mask=~domain),cmap='magma',vmin=0,vmax=30)
            ax.set_title(f'{title} (mm)');fig.colorbar(shown,ax=ax,shrink=.7)
        for ax in axes[row]:ax.set_xlim(-.5,223.5);ax.set_ylim(223.5,-.5);ax.axis('off')
    fig.suptitle('Fixed rank0 first-heavy example; training-partition holdout, 10-degree reference perturbation\n'
                 'Original target masks; no confidence filtering; error colors capped at 30 mm',fontsize=11)
    fig.savefig(root/'fixed_flow_recovery.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
