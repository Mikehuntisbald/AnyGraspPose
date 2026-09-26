"""Render saved teacher-only audits; no access to training or checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    before=a.root/'supervision_audit_v41_r3';after=a.root/'supervision_audit_v41_filtered'
    old=json.loads((before/'audit.json').read_text());new=json.loads((after/'audit.json').read_text())
    assert old['completed'] and new['completed']
    total=kept=proxy_count=proxy_behind=0
    for i,(b,c) in enumerate(zip(old['cases'],new['cases'])):
        assert (b['stream'],b['frame'])==(c['stream'],c['frame'])
        x=np.load(before/f'case{i:02d}.npz');y=np.load(after/f'case{i:02d}.npz')
        assert np.array_equal(x['proxy'],y['proxy'])
        assert np.array_equal(x['target_depth'],y['target_depth'])
        assert np.array_equal(x['target_xyz'],y['target_xyz'])
        total+=int(x['real'].sum());kept+=int(y['real'].sum())
        gap=(x['sensor_depth']-x['cad_depth'])*1000
        proxy_count+=int(x['proxy'].sum());proxy_behind+=int((x['proxy']&(x['sensor_depth']>0)&(gap>30)).sum())
        fig,axes=plt.subplots(2,4,figsize=(15,8))
        for ax,key in zip(axes[0,:3],('rgb','student','rendered_rgb')):
            ax.imshow(x[key].transpose(1,2,0).clip(0,1));ax.set_title(key)
        ownership=np.zeros((224,224,3));ownership[y['real']]=[.1,1,.2];ownership[y['proxy']]=[.1,.4,1]
        ownership[x['real']&~y['real']]=[1,.2,.1]
        axes[0,3].imshow(ownership);axes[0,3].set_title('green real / blue CAD / red excluded')
        dep=x['cad_depth'][x['silhouette']];low,high=np.quantile(dep,[.01,.99]);low-=.03;high+=.03
        for ax,key in zip(axes[1,:2],('sensor_depth','cad_depth')):
            im=ax.imshow(np.where(x[key]>0,x[key],np.nan),vmin=low,vmax=high,cmap='viridis');ax.set_title(key+' (m)');fig.colorbar(im,ax=ax,fraction=.05)
        for ax,mask,title in zip(axes[1,2:],(x['real'],y['real']),('before','after')):
            im=ax.imshow(np.where(mask,gap,np.nan),vmin=-40,vmax=40,cmap='coolwarm');ax.set_title(title+': sensor - CAD (mm)');fig.colorbar(im,ax=ax,fraction=.05)
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(f"Case {i}: object {b['object_id']} | frame {b['frame']} | targets retained unchanged; excluded pixels ignored")
        fig.tight_layout();fig.savefig(a.root/f'case{i:02d}.png',dpi=120);plt.close(fig)
    result=dict(frames=len(old['cases']),objects=sorted({r['object_id'] for r in old['cases']}),
                real_pixels_before=total,real_pixels_after=kept,retained_fraction=kept/total,
                unchanged_proxy_pixels=proxy_count,proxy_measured_behind_cad_over30mm=proxy_behind,
                native_bytes_and_pose_exact=all(r['native_bytes_exact'] and r['native_pose_exact'] for r in old['cases']),
                original_target_values_unchanged=True,maximum_supervised_gap_before_mm=max(r['supervised_real_sensor_cad_gap_mm'].get('maximum',0) for r in old['cases']),
                maximum_supervised_gap_after_mm=max(r['supervised_real_sensor_cad_gap_mm'].get('maximum',0) for r in new['cases']),
                analytic=old['analytic'],training=False,heldout_access=False)
    (a.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
