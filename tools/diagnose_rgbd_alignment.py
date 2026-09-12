"""Read-only RGB-D/pose audit using the project's and native FP renderers.

Uses existing oracle diagnostic selections, never reads hand arrays, and writes
only to the requested diagnostic output directory. No training or pose correction.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from lip.geometry.renderer import Renderer


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = root / 'cache/raw_full_20260910'
    index = root / 'cache/dexycb_s0'
    source = root / 'runs/fp_transition_analysis_21530_31000/gt_initialization_diagnostic.json'
    probe = json.loads(source.read_text())
    streams = {s['stream_id']: s for s in map(json.loads, (index/'streams.jsonl').read_text().splitlines())}
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.cuda.set_device(0)
    sys.path.insert(0, str(root/'third_party/FoundationPose'))
    from Utils import nvdiffrast_render, make_mesh_tensors
    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext(device='cuda:0')
    renderer = Renderer('cuda')
    rows = []
    with torch.no_grad():
        for r in probe['rows']:
            s = streams[r['stream_id']]
            base = raw/s['relative_dir']
            frame = r['frame_index']
            rgb = cv2.cvtColor(cv2.imread(str(base/f'color_{frame:06d}.jpg')), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(base/f'aligned_depth_to_color_{frame:06d}.png'), -1).astype('f4') * .001
            with np.load(base/f'labels_{frame:06d}.npz') as z:
                target = z['seg'] == s['object_id']
                label_pose = np.eye(4, dtype='f4')
                label_pose[:3] = z['pose_y'][s['object_index_in_sequence']]
            with np.load(index/s['mesh_cache']) as z:
                mesh = {k: z[k].copy() for k in z.files}
            raw_mesh = trimesh.load(raw/s['mesh_path'], process=False, force='mesh')
            k = np.array(s['intrinsics'], dtype='f4')
            poses = np.array([r['input_GT_pose'], r['refined_pose']], dtype='f4')
            original = poses.copy()
            original[:, :3, 3] -= poses[:, :3, :3] @ mesh['center']
            raw_points_cam = np.asarray(raw_mesh.vertices) @ original[0, :3, :3].T + original[0, :3, 3]
            cached_points_cam = mesh['vertices'] @ poses[0, :3, :3].T + poses[0, :3, 3]
            # Native renderer has a half-pixel convention relative to Renderer.
            native_mesh = make_mesh_tensors(raw_mesh)
            color, native_d, _ = nvdiffrast_render(K=k, H=480, W=640,
                ob_in_cams=torch.as_tensor(original, device='cuda'), mesh_tensors=native_mesh, glctx=ctx, extra={})
            color = color.cpu().numpy()
            native_d = native_d.cpu().numpy()
            shifted_k = k.copy()
            shifted_k[:2, 2] += .5
            _, aligned_d, _ = nvdiffrast_render(K=shifted_k, H=480, W=640,
                ob_in_cams=torch.as_tensor(original[:1], device='cuda'), mesh_tensors=native_mesh, glctx=ctx, extra={})
            aligned_d = aligned_d[0].cpu().numpy()
            project_d, _ = renderer(mesh, torch.as_tensor(poses[0], device='cuda'), torch.as_tensor(k, device='cuda'), 640)
            project_d = project_d[0, :480].cpu().numpy()
            inside = cv2.erode((target & (depth > 0) & (project_d > 0) & (native_d[0] > 0) & (aligned_d > 0)).astype('uint8'), np.ones((5, 5), 'uint8')).astype(bool)
            residual = (depth-native_d[0])*1000
            y, x = np.where((native_d[0]>0) | (native_d[1]>0) | target)
            x0, x1 = max(0,int(x.min())-22), min(640,int(x.max())+23)
            y0, y1 = max(0,int(y.min())-22), min(480,int(y.max())+23)
            crop = np.s_[y0:y1, x0:x1]
            yy = int(np.median(np.where(inside)[0]))
            fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
            axes[0,0].imshow(rgb[crop], extent=(x0,x1,y1,y0))
            axes[0,0].contour(np.arange(x0,x1), np.arange(y0,y1), (native_d[0][crop]>0), levels=[.5], colors=['lime'], linewidths=1)
            axes[0,0].contour(np.arange(x0,x1), np.arange(y0,y1), (native_d[1][crop]>0), levels=[.5], colors=['magenta'], linewidths=1)
            axes[0,0].set_title('RGB | GT: green, FP(GT): magenta')
            for j, title in enumerate(('Native textured GT overlay', 'Native textured FP(GT) overlay')):
                overlay = rgb.astype('f4')/255
                m = native_d[j]>0
                overlay[m] = .5*overlay[m]+.5*color[j][m]
                axes[0,j+1].imshow(overlay[crop], extent=(x0,x1,y1,y0))
                axes[0,j+1].set_title(title)
            axes[1,0].imshow(rgb[crop], extent=(x0,x1,y1,y0))
            im = axes[1,0].imshow(np.ma.masked_where(~inside,residual)[crop], extent=(x0,x1,y1,y0), cmap='coolwarm',vmin=-35,vmax=35)
            axes[1,0].set_title('Observed depth - GT render (mm)\nObject interior only; red = observed farther')
            fig.colorbar(im, ax=axes[1,0], shrink=.8)
            axes[1,0].axhline(yy, color='k',linestyle='--',linewidth=.8)
            for arr,label,c in ((depth,'Observed','black'),(native_d[0],'GT render','green'),(native_d[1],'FP(GT) render','magenta')):
                v=arr[yy,x0:x1].copy()*1000
                v[v==0]=np.nan
                axes[1,1].plot(np.arange(x0,x1),v,label=label,color=c)
            dvals=native_d[:,yy,x0:x1]; dvals=dvals[dvals>0]*1000
            axes[1,1].set_ylim(dvals.min()-30,dvals.max()+70)
            axes[1,1].set_title(f'Depth profile at row {yy}'); axes[1,1].set_xlabel('Image column'); axes[1,1].set_ylabel('Camera Z (mm)'); axes[1,1].legend(fontsize=8)
            axes[1,2].imshow(rgb)
            axes[1,2].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='yellow'))
            axes[1,2].set_title('Full image and crop location')
            for ax in (axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,2]): ax.set_axis_off()
            title=f"{s['object_id']:02d} {Path(s['mesh_path']).parent.name} | {s['camera_serial']} | frame {frame}\nDepth gap median {np.median(residual[inside]):+.2f} mm | FP Z shift {r['signed_error_xyz_mm'][2]:+.2f} mm | GT-to-FP center {r['after']['center_mm']:.2f} mm"
            fig.suptitle(title, fontsize=12)
            fig.tight_layout()
            image_name=f"object_{s['object_id']:02d}.png"
            fig.savefig(out/image_name, dpi=160)
            plt.close(fig)
            # Keep exact arrays to permit follow-up plots without repeated GPU work.
            np.savez_compressed(out/f"object_{s['object_id']:02d}.npz",rgb=rgb,observed_depth=depth,
                native_depth=native_d,native_rgb=color,project_depth=project_d,target=target,interior=inside,
                K=k,pose_centered=poses,pose_original=original)
            row=dict(object_id=s['object_id'],stream_id=s['stream_id'],frame_index=frame,image=image_name,
                pixels=int(inside.sum()),cached_vs_raw_camera_vertices_max_abs_m=float(np.abs(raw_points_cam-cached_points_cam).max()),
                raw_label_vs_probe_original_pose_max_abs=float(np.abs(label_pose-original[0]).max()),
                native_vs_project_median_abs_mm=float(np.median(np.abs(native_d[0][inside]-project_d[inside]))*1000),
                native_pixel_aligned_vs_project_median_abs_mm=float(np.median(np.abs(aligned_d[inside]-project_d[inside]))*1000),
                native_pixel_aligned_vs_project_max_abs_mm=float(np.max(np.abs(aligned_d[inside]-project_d[inside]))*1000),
                observed_minus_native_GT_median_mm=float(np.median(residual[inside])),
                observed_minus_native_GT_median_abs_mm=float(np.median(np.abs(residual[inside]))))
            rows.append(row)
            print(json.dumps(row),flush=True)
    report=dict(completed=True,scope='Same preselected 20 oracle diagnostic frames. No training. Raw data read only. FP poses reused from prior probe.',
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows)
    (out/'renderer_visual_audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
