"""Fixed rank0/first-heavy example; GT markers are display/scoring only."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--steps',type=int,default=100);a=p.parse_args()
    fig=plt.figure(figsize=(15,8));axes=[]
    for row,step in enumerate((0,a.steps)):
        path=a.root/'probe'/f'step{step}'/'rank0'
        g=np.load(path/'heavy_example.npz');c=np.load(path/'correspondences.npz')
        rgb=g['occluded_rgb'].squeeze().transpose(1,2,0)
        ids=np.flatnonzero(c['gt_support'])
        ids=ids[np.linspace(0,len(ids)-1,min(12,len(ids))).astype(int)]
        ax=fig.add_subplot(2,3,row*3+1,projection='3d')
        cloud=c['cad_xyz_m']*1000
        ax.scatter(*cloud.T,c='lightgray',s=2)
        for i in ids:
            color=plt.cm.tab20(int(i)%20);ax.scatter(*cloud[i],color=color,s=20);ax.text(*cloud[i],str(int(c['cad_ids'][i])),fontsize=6)
        ax.set_title(f'Step {step}: CAD points (mm)');ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
        ax=fig.add_subplot(2,3,row*3+2);ax.imshow(np.clip(rgb,0,1))
        for i in ids:
            p0=c['gt_uv_crop'][i];p1=c['flow_uv_crop' if 'flow_uv_crop' in c else 'uv_crop'][i];color=plt.cm.tab20(int(i)%20)
            ax.scatter(*p0,color=color,marker='o',s=22);ax.scatter(*p1,color=color,marker='x',s=30)
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color=color,linewidth=.8)
            ax.text(*p0,str(int(c['cad_ids'][i])),fontsize=6,color='white',bbox=dict(facecolor='black',alpha=.4,pad=.3))
        ax.set_title('Same CAD IDs: circle=GT, cross=prediction');ax.set_xlim(0,223);ax.set_ylim(223,0);ax.axis('off')
        ax=fig.add_subplot(2,3,row*3+3)
        mask=(g['real_mask']|g['proxy_mask']).squeeze()
        error=np.linalg.norm(g['predicted_xyz']-g['cad_target_xyz'],axis=1).squeeze()*float(g['diameter'])*1000
        image=ax.imshow(np.where(mask,error,np.nan),vmin=0,vmax=30,cmap='magma')
        ax.set_title('Canonical XYZ error on ORIGINAL masks (mm)');ax.axis('off');fig.colorbar(image,ax=ax,shrink=.8)
    fig.suptitle('Fixed first heavy example; point subset chosen from GT support for display only. No GT used in inference.',fontsize=10)
    fig.tight_layout();fig.savefig(a.root/'cad_points_in_image.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
