"""Render saved fixed-heavy-case arrays; no model or target recomputation."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--rank',type=int,default=0)
    a=p.parse_args();root=Path(a.root)
    data=[(name,np.load(root/arm/'probe'/f'step{step}'/f'rank{a.rank}'/'heavy_example.npz'))
          for name,arm,step in [('Source','control',0),('Geometry-only DPT','control',100),('CAD transport','transport',100)]]
    fig,axes=plt.subplots(3,7,figsize=(21,9),layout='constrained')
    for row,(name,d) in enumerate(data):
        mask=(d['real_mask']|d['proxy_mask'])[0,0]
        xyz_error=np.linalg.norm(d['predicted_xyz'][0]-d['target_xyz'][0],axis=0)*float(d['diameter'])*1000
        depth_error=np.abs(d['predicted_depth'][0,0]-d['target_depth'][0,0])*1000
        xyz=np.moveaxis(d['predicted_xyz'][0],0,-1)+.5
        xyz=np.where(mask[...,None],xyz,1.)
        truth=np.moveaxis(d['target_xyz'][0],0,-1)+.5
        truth=np.where(mask[...,None],truth,1.)
        images=[np.moveaxis(d['rgb'][0],0,-1),np.moveaxis(d['occluded_rgb'][0],0,-1),truth,xyz,
                np.where(mask,xyz_error,np.nan),np.where(mask,depth_error,np.nan),d['gate'][0,0]]
        for col,image in enumerate(images):
            options=dict(cmap='magma',vmin=0,vmax=40) if col in (4,5) else (dict(cmap='viridis',vmin=0,vmax=1) if col==6 else {})
            artist=axes[row,col].imshow(np.clip(image,0,1) if col<4 else image,**options)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['Original real RGB','After textured occlusion','Target canonical XYZ','Predicted canonical XYZ','XYZ error (mm)','Depth error (mm)','Transport gate'][col])
            if col==0:axes[row,col].set_ylabel(name)
            if row==2 and col>=4:fig.colorbar(artist,ax=axes[:,col],shrink=.65)
    fig.suptitle('Fixed training-partition holdout: identical input/targets; white pixels are outside eligible real/proxy scoring regions')
    target=root/f'heavy_case_rank{a.rank}.png';fig.savefig(target,dpi=150);plt.close(fig)


if __name__=='__main__':main()
