"""Check mesh simplification and pixel convention on saved alignment audit frames."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import trimesh
from lip.geometry.renderer import Renderer


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.out.resolve();raw=root/'cache/raw_full_20260910'
    selected=json.loads((root/'runs/fp_transition_analysis_21530_31000/gt_initialization_diagnostic.json').read_text())['rows']
    torch.set_num_threads(2);torch.cuda.set_device(0);renderer=Renderer('cuda')
    sys.path.insert(0,str(root/'third_party/FoundationPose'))
    from Utils import nvdiffrast_render,make_mesh_tensors
    import nvdiffrast.torch as dr
    ctx=dr.RasterizeCudaContext(device='cuda:0');rows=[]
    with torch.no_grad():
        for r in selected:
            oid=r['object_id'];path=raw/r['mesh_path']
            with np.load(out/f'object_{oid:02d}.npz') as z:
                observed=z['observed_depth'].copy();project=z['project_depth'].copy();inside=z['interior'].copy()
                K=z['K'].copy();pose=z['pose_original'][0].copy()
            mesh=trimesh.load(path,process=False,force='mesh')
            shifted=K.copy();shifted[:2,2]+=.5
            _,native,_=nvdiffrast_render(K=shifted,H=480,W=640,ob_in_cams=torch.tensor(pose[None],device='cuda'),mesh_tensors=make_mesh_tensors(mesh),glctx=ctx,extra={})
            native=native[0].cpu().numpy();mask=inside&(native>0)
            err=np.abs((native-project)[mask])*1000
            row=dict(object_id=oid,renderer_diff_median_mm=float(np.median(err)),renderer_diff_p99_mm=float(np.percentile(err,99)),
                renderer_diff_max_mm=float(err.max()),renderer_diff_over_1mm_pixels=int((err>1).sum()),renderer_comparison_pixels=len(err))
            if oid in [5,7,14,15,21]:
                path_hi=path.with_name('textured.obj')
                hi=trimesh.load(path_hi,process=False,force='mesh',skip_materials=True)
                # Render raw uncentered mesh with original camera pose.
                d,_=renderer(dict(vertices=np.array(hi.vertices,dtype='f4'),faces=np.array(hi.faces,dtype='i4')),torch.tensor(pose,device='cuda'),torch.tensor(K,device='cuda'),640)
                d=d[0,:480].cpu().numpy();mask=inside&(d>0)
                row.update(high_res_vertices=len(hi.vertices),simple_vertices=len(mesh.vertices),
                    high_vs_simple_depth_median_abs_mm=float(np.median(np.abs((d-project)[mask]))*1000),
                    observed_minus_high_res_median_mm=float(np.median((observed-d)[mask])*1000),
                    observed_minus_simple_median_mm=float(np.median((observed-project)[mask])*1000),
                    simple_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),high_res_sha256=hashlib.sha256(path_hi.read_bytes()).hexdigest())
            rows.append(row);print(json.dumps(row),flush=True)
    (out/'mesh_depth_hypotheses.json').write_text(json.dumps(dict(completed=True,
        scope='20 renderer cross-checks; original high-resolution textured.obj vs textured_simple.obj on objects 5,7,14,15,21. Raw mesh and data unchanged.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows),indent=2))


if __name__=='__main__':main()
