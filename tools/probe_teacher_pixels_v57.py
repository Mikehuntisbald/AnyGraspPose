"""Independent geometric teacher roundtrip; no learned forward or optimization."""
import argparse,json,sys
import numpy as np
from pathlib import Path
import torch,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,build_teachers
from lip.unified.execution_speed import crop_images_fast
from lip.unified.cad_transport import pixel_grid
from lip.geometry.so3 import update


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-strict',action='store_true');a=p.parse_args()
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(42)
    c=yaml.safe_load(Path('configs/jepa/flow_reconstruction_v56.yaml').read_text());m=build_model(c).requires_grad_(False).eval()
    f=Factory(c,m,make_store(c,m));e,(truth,masks)=f.sample(a.seed,frames=1)
    d=float(e.mesh['diameter']);rot=truth.new_tensor([[0.,0.,torch.pi/18]])
    base=update(truth[:1],rot,torch.zeros_like(rot),truth.new_tensor([d]))
    scene=prepare_scene(e.rgb[0],e.depth[0],base[0],e.mesh,e.k,e.times[0],e.stream,e.cad,f.renderer,fast=True)
    t=build_teachers(m.ema_teacher,[scene],truth[:1],masks[:1],[torch.zeros_like(scene.bounds)],f.renderer,
        real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
    camera=torch.einsum('bij,bjhw->bihw',truth[:1,:3,:3].float(),t.cad_geometry_xyz.float()*d)+truth[:1,:3,3,None,None].float()
    projection=torch.einsum('bij,bjhw->bihw',scene.k_crop[None].float(),camera)
    uv=projection[:,:2]/projection[:,2:3]
    mask=-F.max_pool2d(-t.cad_geometry_valid.float(),5,1,2)>.999
    epe=(uv-pixel_grid(1,224,224,'cuda')).norm(dim=1,keepdim=True)
    z_error=(camera[:,2:3]-t.cad_geometry_depth_m).abs()
    ray_error=(t.camera_rays*t.cad_geometry_depth_m-camera).norm(dim=1,keepdim=True)
    visible=(crop_images_fast(masks[:1].float(),scene.affine,mode='nearest')>.5)&scene.bounds
    real=mask&visible&(scene.depth>0)&torch.isfinite(scene.depth)
    gap=(scene.depth-t.cad_geometry_depth_m)[real]*1000
    sid=e.stream.split('|')[0];stream=f.streams[sid];frame=e.training_window['first_frame']
    with np.load(f.root/stream['relative_dir']/f'labels_{frame:06d}.npz',allow_pickle=False) as archive:
        original=truth.new_tensor(archive['pose_y'][stream['object_index_in_sequence']])
    center=truth.new_tensor(np.asarray(e.mesh['center']));vertices=e.cad['appearance']['vertices']
    original_camera=(vertices+center)@original[:3,:3].T+original[:3,3]
    centered_camera=vertices@truth[0,:3,:3].T+truth[0,:3,3]
    centering_error=(original_camera-centered_camera).abs().max()*1000
    cached_pose_error=float((torch.as_tensor(f.pose_labels[sid][frame],device=truth.device)[:3]-original).abs().max())
    geo_vertices=e.mesh['vertices'].float()
    bounds_error=max(float((geo_vertices.amin(0)-vertices.amin(0)).abs().max()),float((geo_vertices.amax(0)-vertices.amax(0)).abs().max()))*1000
    result=dict(seed=a.seed,stream=e.stream,learned_forward=False,training=False,teacher_pixels=int(mask.sum()),
        pixel_reprojection_max=float(epe[mask].max()),pixel_reprojection_mean=float(epe[mask].mean()),
        camera_z_max_error_mm=float(z_error[mask].max()*1000),camera_xyz_ray_max_error_mm=float(ray_error[mask].max()*1000),
        visible_real_pixels=int(real.sum()),real_minus_cad_depth_mean_mm=float(gap.mean()),
        real_cad_depth_absolute_mean_mm=float(gap.abs().mean()),real_cad_depth_absolute_p90_mm=float(gap.abs().quantile(.9)),
        object_id=stream['object_id'],mesh_path=stream['mesh_path'],frame=frame,
        centering_roundtrip_max_mm=float(centering_error),native_annotation_cache_max_difference=cached_pose_error,
        geometry_appearance_bounds_max_difference_mm=bounds_error,
        scope='Teacher raster/camera/crop roundtrip; does not certify real-scene annotation or calibration accuracy.')
    result['strict_numeric_gate_passed']=result['pixel_reprojection_max']<.002 and result['camera_xyz_ray_max_error_mm']<.01
    result['strict_numeric_limits']=dict(pixel_reprojection_max=.002,camera_xyz_ray_max_error_mm=.01)
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    render=f.renderer(scene.cad['appearance'],truth[0],scene.k_crop,224)
    np.savez_compressed(a.out.with_suffix('.npz'),rgb=scene.rgb.cpu().numpy(),cad_rgb=render['rgb'].cpu().numpy(),
        real_depth=scene.depth.cpu().numpy(),cad_depth=t.cad_geometry_depth_m.cpu().numpy(),visible=visible.cpu().numpy(),
        domain=real.cpu().numpy(),cad_mask=t.cad_geometry_valid.cpu().numpy(),k=scene.k_crop.cpu().numpy(),gt_pose=truth[0].cpu().numpy())
    assert cached_pose_error<1e-6 and float(centering_error)<.001 and bounds_error<.001,result
    if a.require_strict:assert result['strict_numeric_gate_passed'],result


if __name__=='__main__':main()
