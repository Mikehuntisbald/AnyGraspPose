import argparse,json,os
from pathlib import Path
import cv2,numpy as np,torch
from lip.data.index import read_frame
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import center_pose,original_pose
from lip.geometry.crop import crop_matrix,project
from lip.evaluate import overlay
p=argparse.ArgumentParser();p.add_argument('--data-root',default=os.getenv('DEX_YCB_DIR'));p.add_argument('--index',default='cache/dexycb_s0');p.add_argument('--num-samples',type=int,default=32);p.add_argument('--out',default='runs/geometry');p.add_argument('--device',default='cuda');a=p.parse_args()
if not a.data_root:p.error('DEX_YCB_DIR required')
index=Path(a.index);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);audit=json.loads((index/'audit.json').read_text())
streams=[json.loads(x) for x in (index/'streams.jsonl').read_text().splitlines() if json.loads(x)['split']=='train']
rng=np.random.default_rng(42);renderer=Renderer(a.device);rows=[]
objects=sorted(set(s['object_id'] for s in streams));groups={o:[s for s in streams if s['object_id']==o] for o in objects}
for i in range(a.num_samples):
 group=groups[objects[i%len(objects)]];s=group[int(rng.integers(len(group)))]
 with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
 with np.load(index/s['pose_cache']) as z:
  j=int(rng.integers(len(z['frames'])));frame=int(z['frames'][j]);orig=torch.tensor(z['poses'][j],device=a.device)
 center=torch.tensor(mesh['center'],device=a.device);gt=center_pose(orig,center);k=torch.tensor(s['intrinsics'],device=a.device)
 rd,xyz=renderer(mesh,gt,k,640);rd=rd[0,:480].cpu().numpy()
 rgb,depth=read_frame(a.data_root,s,frame,audit['depth_scale_to_m'])
 with np.load(Path(a.data_root)/s['relative_dir']/f'labels_{frame:06d}.npz') as z:mask=z['seg']==s['object_id']
 valid=mask&(rd>0)&(depth[0]>0);res=np.abs(rd[valid]-depth[0][valid])
 v=torch.tensor(mesh['points'],device=a.device);cam=v@gt[:3,:3].T+gt[:3,3];uv=project(cam,k)
 mat,kc=crop_matrix(torch.tensor(mesh['vertices'],device=a.device),gt,k)
 uv1=project(cam,kc);mapped=torch.cat((uv,torch.ones_like(uv[:,:1])),-1)@mat.T
 row=dict(object_id=s['object_id'],stream=s['stream_id'],frame=frame,visible_pixels=int(valid.sum()),mesh_diameter_m=float(mesh['diameter']),
          center_camera_m=gt[:3,3].cpu().tolist(),roundtrip_max=float((original_pose(gt,center)-orig).abs().max()),
          crop_projection_max=float((uv1-mapped[:,:2]).abs().max()),
          depth_abs_median_m=float(np.median(res)) if len(res) else None,
          depth_abs_p95_m=float(np.quantile(res,.95)) if len(res) else None,K=k.cpu().tolist(),A=mat.cpu().tolist(),K_crop=kc.cpu().tolist())
 rows.append(row);overlay(out/f'{i:03d}_overlay.jpg',rgb,gt.cpu().numpy(),gt.cpu().numpy(),mesh,k.cpu().numpy())
 image=cv2.imread(str(out/f'{i:03d}_overlay.jpg'))
 corners=torch.tensor([[0.,0,1],[223.,0,1],[223.,223.,1],[0,223.,1]],device=a.device)@torch.linalg.inv(mat).T
 cv2.polylines(image,[corners[:,:2].cpu().numpy().astype('int32')],True,(0,255,255),2)
 uvcenter=project(gt[:3,3][None],k)[0].cpu().numpy().astype(int)
 cv2.circle(image,tuple(uvcenter),5,(255,0,255),-1)
 cv2.putText(image,f'center z={float(gt[2,3]):.3f} m; diameter={float(mesh["diameter"]):.3f} m',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
 cv2.imwrite(str(out/f'{i:03d}_overlay.jpg'),image)
 np.savez_compressed(out/f'{i:03d}_depth.npz',observed_m=depth[0],rendered_m=rd,visible_target=valid)
 vis=np.concatenate([np.clip(depth[0]/1.5*255,0,255).astype('uint8'),np.clip(rd/1.5*255,0,255).astype('uint8'),np.clip(np.abs(depth[0]-rd)/.05*255,0,255).astype('uint8')],axis=1)
 cv2.imwrite(str(out/f'{i:03d}_depth.png'),cv2.applyColorMap(vis,cv2.COLORMAP_TURBO))
validrows=[r for r in rows if r['visible_pixels']>=50]
passed=(len(validrows)>=max(1,a.num_samples//2) and all(r['roundtrip_max']<1e-5 and r['crop_projection_max']<.002 for r in rows)
        and np.median([r['depth_abs_median_m'] for r in validrows])<.02)
report=dict(passed=bool(passed),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],units=dict(depth=audit['depth_scale_to_m'],pose=audit['pose_scale_to_m'],mesh=audit['mesh_scale_to_m']),criteria='>=half samples have >=50 visible pixels; median of per-sample depth medians <20mm; roundtrip <1e-5; crop <.002px',samples=rows)
(out/'geometry.json').write_text(json.dumps(report,indent=2));(index/'geometry_gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=bool(passed),samples=len(rows),depth_valid_samples=len(validrows))))
if not passed:raise SystemExit(2)
