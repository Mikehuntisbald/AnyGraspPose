"""Compare background table depth between calibrated cameras, without object GT poses."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lip.data.index import read_yaml, CAMERAS


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    raw=root/'cache/raw_full_20260910'
    source=root/'runs/fp_transition_analysis_21530_31000/gt_initialization_diagnostic.json'
    chosen=json.loads(source.read_text())['rows']; y,x=np.mgrid[:480,:640]; pixel=np.stack((x,y,np.ones_like(x)),axis=-1)
    rows=[];paired=[];maps=[];cv2.setNumThreads(1)
    for si,sel in enumerate(chosen):
        physical='/'.join(sel['stream_id'].split('/')[:2]);fi=sel['frame_index']
        meta=read_yaml(raw/physical/'meta.yml')
        ext=read_yaml(raw/'calibration'/f"extrinsics_{meta['extrinsics']}"/'extrinsics.yml')['extrinsics']
        tag=np.eye(4);tag[:3]=np.array(ext['apriltag']).reshape(3,4);invtag=np.linalg.inv(tag)
        cells=[]
        for ci,camera in enumerate(CAMERAS):
            base=raw/physical/camera
            d=cv2.imread(str(base/f'aligned_depth_to_color_{fi:06d}.png'),-1).astype('f4')*.001
            rgb=cv2.imread(str(base/f'color_{fi:06d}.jpg'))
            # Only background segmentation is used; pose_y and hand arrays are not read.
            with np.load(base/f'labels_{fi:06d}.npz') as z: background=cv2.erode((z['seg']==0).astype('uint8'),np.ones((9,9),'uint8')).astype(bool)
            intr=read_yaml(raw/'calibration/intrinsics'/f'{camera}_640x480.yml')['color']
            k=np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1]])
            E=np.eye(4);E[:3]=np.array(ext[camera]).reshape(3,4);E=invtag@E
            cloud=(pixel@np.linalg.inv(k).T)*d[...,None];cloud=cloud@E[:3,:3].T+E[:3,3]
            mask=background&(d>0)&(rgb.max(-1)<65)&(cloud[...,0]>.15)&(cloud[...,0]<1.05)&(cloud[...,1]>.05)&(cloud[...,1]<.55)&(np.abs(cloud[...,2])<.04)
            pts=cloud[mask]; assert len(pts)>500,(physical,camera,len(pts))
            bins=np.floor((pts[:,:2]-[.15,.05])/.02).astype(int); cell_id=bins[:,1]*45+bins[:,0]
            cell=np.full((25*45,),np.nan)
            order=np.argsort(cell_id); ids=cell_id[order]; zz=pts[order,2]*1000
            edges=np.r_[0,np.flatnonzero(np.diff(ids))+1,len(ids)]
            for l,r in zip(edges[:-1],edges[1:]):
                if r-l>=5:cell[ids[l]]=np.median(zz[l:r])
            cells.append(cell)
            rows.append(dict(selection=si,camera=camera,physical_sequence=physical,frame_index=fi,pixels=len(pts),cells=int(np.isfinite(cell).sum()),median_height_mm=float(np.nanmedian(cell))))
            if sel['object_id']==15:
                preview=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB).copy();preview[mask]=(.5*preview[mask]+.5*np.array([0,255,200])).astype('uint8')
                cv2.imwrite(str(out/f'table_mask_{camera}.png'),cv2.cvtColor(preview,cv2.COLOR_RGB2BGR))
        cells=np.array(cells);maps.append(cells.reshape(8,25,45))
        for i,c1 in enumerate(CAMERAS):
            for j,c2 in enumerate(CAMERAS):
                if j<=i:continue
                valid=np.isfinite(cells[i])&np.isfinite(cells[j])
                if valid.sum()>=20:paired.append(dict(selection=si,camera_a=c1,camera_b=c2,common_cells=int(valid.sum()),b_minus_a_height_mm=float(np.median(cells[j,valid]-cells[i,valid]))))
        print(json.dumps(dict(completed_selections=si+1)),flush=True)
    maps=np.array(maps)
    report=dict(completed=True,scope='Read-only background table diagnostic. No object GT poses or hand arrays. Background segmentation, max RGB <65, apriltag XY [.15,1.05] x [.05,.55] m, |Z|<.04 m; 20mm cells with >=5 points, pairs require >=20 shared cells. Absolute table height is not assumed to be GT; only common-cell camera differences are interpreted.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,paired=paired)
    (out/'table_consistency.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'table_height_maps.npz',values=maps,cameras=np.array(CAMERAS))
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for ci,ax in enumerate(axes.flat):
        med=np.nanmedian(maps[:,ci],axis=0)
        im=ax.imshow(med,origin='lower',extent=(.15,1.05,.05,.55),cmap='coolwarm',vmin=-20,vmax=20)
        ax.set_title(CAMERAS[ci]);ax.set_xlabel('Apriltag X (m)');ax.set_ylabel('Apriltag Y (m)')
    fig.colorbar(im,ax=axes,shrink=.75,label='Observed background height in common frame (mm)')
    fig.suptitle('Same table, different cameras | 20 fixed frames | No object GT pose used')
    fig.savefig(out/'table_camera_height.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
