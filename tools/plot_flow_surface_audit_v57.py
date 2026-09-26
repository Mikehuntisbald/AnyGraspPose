"""Fixed rank0 first-heavy sample; reconstruction accuracy, not pose scores."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    path=a.root/'angle10/rank0'
    raw=np.load(path/'heavy_example.npz');candidates=np.load(path/'surface_audit_example.npz')
    image=raw['occluded_rgb'][0].transpose(1,2,0);domain=(raw['real_mask']|raw['proxy_mask'])[0,0]
    fig,axes=plt.subplots(4,2,figsize=(9,14),layout='constrained')
    for row,name in enumerate(('baseline','rigid','metric_recovered','oracle_cad')):
        xyz=raw['predicted_xyz'] if name=='baseline' else candidates[name+'_xyz']
        depth=raw['predicted_depth'] if name=='baseline' else candidates[name+'_depth']
        errors=[np.linalg.norm(xyz-raw['cad_target_xyz'],axis=1)[0]*float(raw['diameter'])*1000,
                np.abs(depth-raw['target_depth'])[0,0]*1000]
        for col,(error,title) in enumerate(zip(errors,('Canonical XYZ error','Depth error'))):
            ax=axes[row,col];ax.imshow(image)
            plot=ax.imshow(np.ma.array(error,mask=~domain),vmin=0,vmax=30,cmap='magma')
            ax.set_title(f'{name}: {title} (mm)');ax.axis('off');fig.colorbar(plot,ax=ax,shrink=.7)
    fig.suptitle('Fixed rank0 first-heavy sample; colors capped at 30 mm\nOracle CAD uses GT and is a supervision control, not a prediction',fontsize=12)
    fig.savefig(a.root/'fixed_surface_audit.png',dpi=150);plt.close(fig)
    if (a.root/'teacher_pixel_roundtrip.npz').exists():
        d=np.load(a.root/'teacher_pixel_roundtrip.npz');rgb=d['rgb'][0].transpose(1,2,0);cad=d['cad_rgb'].transpose(1,2,0)
        mask=d['cad_mask'][0,0];domain=d['domain'][0,0];real=d['real_depth'][0,0];render=d['cad_depth'][0,0]
        fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
        for ax,img,title in zip(axes[0],(rgb,cad,np.where(mask[...,None],.5*rgb+.5*cad,rgb)),('Original RGB','GT CAD RGB','50% GT CAD overlay')):
            ax.imshow(img);ax.set_title(title);ax.axis('off')
        bounds=np.percentile(np.r_[real[domain],render[domain]],[1,99])
        for ax,value,title in zip(axes[1,:2],(real,render),('Measured Z (m)','GT CAD Z (m)')):
            ax.imshow(rgb);p=ax.imshow(np.ma.array(value,mask=~domain),vmin=bounds[0],vmax=bounds[1],cmap='viridis');ax.set_title(title);ax.axis('off');fig.colorbar(p,ax=ax,shrink=.7)
        ax=axes[1,2];ax.imshow(rgb);p=ax.imshow(np.ma.array((real-render)*1000,mask=~domain),vmin=-30,vmax=30,cmap='coolwarm');ax.set_title('Measured minus GT CAD Z (mm)');ax.axis('off');fig.colorbar(p,ax=ax,shrink=.7)
        fig.suptitle('Fixed supervision check: original visible surface, no learned prediction or added occluder')
        fig.savefig(a.root/'teacher_alignment.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
