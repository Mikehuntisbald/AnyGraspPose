"""Independent analytic and native-data audit; no model or optimizer is loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lip.unified.renderer import FullTextureRenderer
from lip.unified.features import prepare_scene, build_teachers
from lip.geometry.appearance_renderer import load_appearance_mesh
from lip.geometry.crop import crop_images
from lip.geometry.so3 import center_pose


def stats(value):
    x = value.detach().float().flatten()
    return dict(n=x.numel(), mean=float(x.mean()), p50=float(x.median()),
                p95=float(torch.quantile(x, .95)), maximum=float(x.max())) if x.numel() else dict(n=0)


def independent_camera(depth, k):
    # Solve K X = pixel*z in float64; independent of production inverse/ray helper.
    y, x = torch.meshgrid(torch.arange(224, device=k.device), torch.arange(224, device=k.device), indexing='ij')
    pixel = torch.stack((x, y, torch.ones_like(x))).double().reshape(3, -1)
    return torch.linalg.solve(k.double(), pixel).reshape(3,224,224)*depth.double()


def renderer_errors(render, pose, k):
    mask = render['mask']
    xyz = render['xyz'].double().reshape(3,-1)
    camera = (pose[:3,:3].double()@xyz+pose[:3,3,None].double()).reshape(3,224,224)
    actual = independent_camera(render['depth'], k)
    return stats((camera-actual).norm(dim=0)[mask]*1000)


def analytic(renderer):
    from scipy.spatial.transform import Rotation
    device='cuda'
    v=torch.tensor([[-.3,-.3,0],[.3,-.3,0],[.3,.3,0],[-.3,.3,0]],device=device)
    mesh=dict(vertices=v,faces=torch.tensor([[0,1,2],[0,2,3]],device=device,dtype=torch.int32),
              normals=torch.tensor([[0.,0,1]],device=device).expand(4,-1).contiguous(),
              uv=torch.tensor([[0.,0],[1,0],[1,1],[0,1]],device=device),texture=torch.ones(1,8,8,3,device=device))
    pose=torch.eye(4,device=device);pose[:3,:3]=torch.tensor(Rotation.from_euler('xyz',[17,-23,11],degrees=True).as_matrix(),device=device)
    pose[:3,3]=torch.tensor([.025,-.031,.8],device=device)
    k=torch.tensor([[260.,0,117.2],[0,290.,103.7],[0,0,1]],device=device)
    # Realistic nonidentity crop, checked separately from render projection.
    affine=torch.tensor([[.83,0,13.2],[0,.83,25.1],[0,0,1]],device=device);kc=affine@k
    r=renderer(mesh,pose,kc,224)
    ray=independent_camera(torch.ones_like(r['depth']),kc)
    normal=pose[:3,2].double()
    expected_z=(normal@pose[:3,3].double())/(ray*normal[:,None,None]).sum(0)
    plane_error=(expected_z-r['depth'][0]).abs()[r['mask']]*1000
    camera_error=renderer_errors(r,pose,kc)
    # Raster edge quantization is submillimetric; interior is checked separately.
    interior=-torch.nn.functional.max_pool2d(-r['mask'][None,None].float(),5,1,2)[0,0]>.999
    interior_error=(expected_z-r['depth'][0]).abs()[interior]*1000
    assert float(plane_error.max())<.1,stats(plane_error)
    assert float(interior_error.max())<.002,stats(interior_error)
    assert camera_error['maximum']<.1,camera_error
    # Affine map and camera matrix agree for analytically known object points.
    camera=v.double()@pose[:3,:3].double().T+pose[:3,3].double()
    p=camera@k.double().T;p=p/p[:,2:]
    direct=p@affine.double().T
    via_k=camera@kc.double().T;via_k=via_k/via_k[:,2:]
    assert float((direct-via_k).abs().max())<1e-4
    center=torch.tensor([.03,-.02,.04],device=device)
    centered=center_pose(pose,center)
    assert torch.allclose((v-center)@centered[:3,:3].T+centered[:3,3],v@pose[:3,:3].T+pose[:3,3],atol=1e-7)
    # Independently predict nearest sampling source pixels, including crop offsets.
    y,x=torch.meshgrid(torch.arange(224,device=device),torch.arange(224,device=device),indexing='ij')
    grid=(y*224+x).float()[None,None]
    sampled=crop_images(grid,affine,mode='nearest')[0,0]
    sx=((x-affine[0,2])/affine[0,0]).round().long();sy=((y-affine[1,2])/affine[1,1]).round().long()
    inside=(sx>=0)&(sx<224)&(sy>=0)&(sy<224)
    assert torch.equal(sampled[inside],(sy*224+sx)[inside].float())
    return dict(plane_depth_error_mm=stats(plane_error),plane_interior_error_mm=stats(interior_error),renderer_xyz_camera_error_mm=camera_error,
                nonidentity_crop=True,centered_pose_equivalence=True,nearest_crop_exact=True)


class AppearanceOnlyStore:
    def get(self,path,mesh):
        return dict(appearance=load_appearance_mesh(path,mesh['center'],'cuda',texture_size=4096))


def real_case(factory,seed,out,index):
    ep,(truth,masks)=factory.sample(seed,frames=1)
    pose=truth[0]
    s=prepare_scene(ep.rgb[0],ep.depth[0],ep.initial,ep.mesh,ep.k,ep.times[0],ep.stream,ep.cad,factory.renderer,fast=True)
    stream=factory.streams[ep.stream.split('|')[0]];frame=round(ep.times[0]*factory.audit['fps'])
    path=factory.root/stream['relative_dir']
    native_depth=cv2.imread(str(path/f'aligned_depth_to_color_{frame:06d}.png'),-1)
    native_rgb=cv2.cvtColor(cv2.imread(str(path/f'color_{frame:06d}.jpg')),cv2.COLOR_BGR2RGB)
    with np.load(path/f'labels_{frame:06d}.npz',allow_pickle=False) as labels:
        raw_mask=labels['seg']==stream['object_id']
        native_pose=torch.eye(4,device='cuda');native_pose[:3]=torch.tensor(labels['pose_y'][stream['object_index_in_sequence']],device='cuda')
    native_centered=center_pose(native_pose,torch.as_tensor(ep.mesh['center'],device='cuda'))
    assert torch.equal(native_centered,pose), 'Native pose labels differ from cached training truth'
    assert np.array_equal(raw_mask,masks[0,0].cpu().numpy())
    assert torch.equal(ep.depth[0,0],torch.tensor(native_depth.astype('float32'),device='cuda')*factory.audit['depth_scale_to_m'])
    assert torch.equal(ep.rgb[0],torch.tensor(native_rgb.transpose(2,0,1),device='cuda').float()/255)
    rng=np.random.default_rng(seed+34001)
    plan=factory.occluders.plan(rng,stream['object_id'],heavy=True,start=0,duration=1)
    occ=plan.render(s,0)
    v=crop_images(masks[0:1].float(),s.affine,mode='nearest')>.5
    saved={key:s.render[key].clone() for key in s.render}
    t=build_teachers(None,[s],truth,masks,[occ.mask],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
    r=factory.renderer(s.cad['appearance'],pose,s.k_crop,224)
    real,proxy=t.geometry_real_weight,t.geometry_proxy_weight
    artificial=v&occ.mask
    assert not (real&proxy).any()
    assert not (real&~artificial).any()
    assert not (proxy&(v|~r['mask'][None,None])).any()
    assert not (t.geometry_weight&v&~occ.mask).any()
    assert torch.equal(t.surface_depth_m[real],s.depth[real])
    assert torch.equal(t.surface_depth_m[proxy],r['depth'][None][proxy])
    assert not (real&((s.depth<=0)|~torch.isfinite(s.depth))).any()
    cam=independent_camera(s.depth[0],s.k_crop)
    independent_xyz=torch.linalg.solve(pose[:3,:3].double(),(cam-pose[:3,3,None,None].double()).reshape(3,-1)).reshape(1,3,224,224)/s.diameter
    real_error=(independent_xyz-t.surface_xyz.double()).norm(dim=1,keepdim=True)[real]*s.diameter*1000
    assert not real_error.numel() or float(real_error.max())<.002
    render_error=renderer_errors(r,pose,s.k_crop)
    assert render_error['p95']<.002,render_error
    mixed_cam=(pose[:3,:3].double()@t.surface_xyz[0].double().reshape(3,-1)*s.diameter+pose[:3,3,None].double()).reshape(3,224,224)
    mixed_z=(mixed_cam[2]-t.surface_depth_m[0,0]).abs()[t.geometry_weight[0,0]]*1000
    # Teacher construction cannot modify the estimated-pose render/observation.
    alternate=truth.clone();alternate[:,0,3]+=.03
    build_teachers(None,[s],alternate,masks,[torch.zeros_like(occ.mask)],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
    assert all(torch.equal(value,s.render[key]) for key,value in saved.items())
    outside=~occ.mask
    assert torch.equal(occ.depth[outside],s.depth[outside])
    assert torch.equal(occ.rgb[outside.expand_as(s.rgb)],s.rgb[outside.expand_as(s.rgb)])
    # Target definition disagreement is diagnostic, not silently counted as noise-free truth.
    gap=(s.depth-r['depth'][None]).abs()*1000
    visible_valid=v&(s.depth>0)&r['mask'][None,None]&s.bounds
    row=dict(seed=seed,stream=ep.stream,frame=frame,object_id=stream['object_id'],diameter_m=s.diameter,
             native_bytes_exact=True,real_pixels=int(real.sum()),proxy_pixels=int(proxy.sum()),
             native_pose_exact=True,render_over_half_mm=render_error['maximum']>=.5,
             unchanged_visible_reconstruction_pixels=int((t.geometry_weight&v&~occ.mask).sum()),
             real_xyz_independent_error_mm=stats(real_error),renderer_camera_error_mm=render_error,
             mixed_xyz_depth_error_mm=stats(mixed_z),visible_sensor_cad_gap_mm=stats(gap[visible_valid]),
             supervised_real_sensor_cad_gap_mm=stats(gap[real]),
             visible_outside_cad_pixels=int((v&~r['mask'][None,None]&s.bounds).sum()),
             gt_teacher_does_not_mutate_student=True,texture_shape=list(s.cad['appearance']['texture'].shape))
    np.savez_compressed(out/f'case{index:02d}.npz',rgb=s.rgb[0].cpu().numpy(),student=occ.rgb[0].cpu().numpy(),
        rendered_rgb=r['rgb'].cpu().numpy(),sensor_depth=s.depth[0,0].cpu().numpy(),cad_depth=r['depth'][0].cpu().numpy(),
        target_depth=t.surface_depth_m[0,0].cpu().numpy(),real=real[0,0].cpu().numpy(),proxy=proxy[0,0].cpu().numpy(),
        visible=v[0,0].cpu().numpy(),added=occ.mask[0,0].cpu().numpy(),silhouette=r['mask'].cpu().numpy(),
        target_xyz=t.surface_xyz[0].cpu().numpy(),cad_xyz=t.cad_geometry_xyz[0].cpu().numpy(),
        geometry_valid_label=t.geometry_valid_label[0,0].cpu().numpy())
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',type=int,default=16);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42);cv2.setNumThreads(0)
    renderer=FullTextureRenderer('cuda')
    result=dict(training=False,model_loaded=False,analytic=analytic(renderer),cases=[])
    (a.out/'analytic.json').write_text(json.dumps(result['analytic'],indent=2))
    from lip.unified.training import Factory
    c=yaml.safe_load(Path(a.config).read_text())
    factory=Factory(c,SimpleNamespace(encoder=None),AppearanceOnlyStore())
    for i in range(a.cases):
        row=real_case(factory,40000000+i,a.out,i);result['cases'].append(row)
        (a.out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
    result.update(completed=True,hard_checks_passed=True,scope='bounded train frames; annotation accuracy not assumed',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'audit.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
