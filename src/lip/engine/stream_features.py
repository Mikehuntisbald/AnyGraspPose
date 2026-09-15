"""One current RGB-D observation, one accepted-pose crop and one CAD render."""
import torch
from torch.nn import functional as F
from lip.geometry.crop import crop_matrix,crop_images,geometry_channels
from lip.geometry.so3 import log


def mesh_to_device(mesh,device):
    return {k:torch.as_tensor(v,device=device) if k in ('vertices','faces','center','diameter','points') else v for k,v in mesh.items()}


@torch.no_grad()
def build_current_features(rgb,depth,base,k,mesh,renderer,timestamp,last_accepted_timestamp,
                           previous_pose=None,previous_timestamp=None,size=224,expansion=2.,
                           beta=1.,epsilon=.1,motion_base=None):
    """No current pose/label/GT inputs. Geometry remains FP32 under autocast."""
    if rgb.ndim!=3 or depth.ndim!=3 or rgb.shape[0]!=3 or depth.shape[0]!=1 or rgb.shape[-2:]!=depth.shape[-2:]:
        raise ValueError('One RGB CHW and depth 1HW frame required')
    device=base.device;h,w=rgb.shape[-2:]
    with torch.autocast(device.type,enabled=False):
        rgb=rgb.to(device).float();depth=depth.to(device).float();base=base.float();k=k.float()
        if not torch.isfinite(rgb).all() or not torch.isfinite(base).all() or not torch.isfinite(k).all():
            raise ValueError('Nonfinite RGB / accepted pose / intrinsics')
        depth_nonfinite=(~torch.isfinite(depth)).any()
        depth=torch.where(torch.isfinite(depth)&(depth>0),depth,0.)
        d=torch.as_tensor(mesh['diameter'],device=device,dtype=torch.float32)
        if not bool(torch.isfinite(d)&(d>0)):raise ValueError('Diameter must be positive meters')
        vertices=torch.as_tensor(mesh['vertices'],device=device,dtype=torch.float32)
        a,kc=crop_matrix(vertices,base,k,size,expansion)
        crop_rgb=crop_images(rgb[None],a,size)[0];crop_depth=crop_images(depth[None],a,size,'nearest')[0]
        rd,xyz=renderer(mesh,base,kc,size)
        geom=geometry_channels(crop_depth[None],rd[None],xyz[None],d,base[2,3])[0]
        fraction=F.adaptive_avg_pool2d((rd>0).float()[None],4)[0,0].flatten()
        reliable=(rd>0).any()
        role=torch.where(reliable,beta*torch.log(epsilon+(1-epsilon)*fraction),torch.zeros_like(fraction))
        role=torch.cat((role,role.new_zeros(1)))
        # Crop cell centers use the same integer-pixel-center convention as crop_images.
        cell=(torch.arange(4,device=device,dtype=torch.float32)+.5)*size/4-.5
        yy,xx=torch.meshgrid(cell,cell,indexing='ij')
        uv=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(16,3)@torch.linalg.inv(a).T
        source_xy=uv[:,:2]/uv.new_tensor([w,h])
        now=torch.as_tensor(timestamp,dtype=torch.float64,device=device)
        last=torch.as_tensor(last_accepted_timestamp,dtype=torch.float64,device=device)
        delta=base.new_zeros(3);motion=base.new_zeros(3);dt=base.new_zeros(1);motion_valid=base.new_zeros(1)
        if previous_pose is not None and previous_timestamp is not None:
            past=torch.as_tensor(previous_timestamp,dtype=torch.float64,device=device)
            if bool(last>past):
                previous_pose=previous_pose.float();accepted=base if motion_base is None else motion_base.float()
                delta=log(accepted[:3,:3]@previous_pose[:3,:3].T)
                motion=(accepted[:3,3]-previous_pose[:3,3])/d;dt=(last-past).float().reshape(1);motion_valid.fill_(1)
        valid_depth=(crop_depth>0).float().mean().reshape(1)
        state=torch.cat((base[:3,:2].T.flatten(),base[:3,3]/d,d.log().reshape(1),
            torch.stack((kc[0,0],kc[1,1],kc[0,2],kc[1,2]))/size,delta,motion,dt,(now-last).float().reshape(1),motion_valid,valid_depth))
        assert state.shape==(24,)
        mean=rgb.new_tensor([.485,.456,.406])[:,None,None];std=rgb.new_tensor([.229,.224,.225])[:,None,None]
        f=dict(rgb=(crop_rgb-mean)/std,geometry=geom,state_input=state,source_xy=source_xy,
               T_base_centered=base,object_diameter_m=d,mesh_center=torch.as_tensor(mesh['center'],device=device,dtype=torch.float32))
        # Source bounds and projection validity are diagnostics, not calibrated failure probabilities.
        camera=vertices@base[:3,:3].T+base[:3,3]
        diag=dict(A=a.clone(),K_crop=kc.clone(),T_base_source=base.clone(),crop_rgb=crop_rgb,render_depth=rd,
            silhouette_token_fraction=fraction,role_bias=role,geometry_reliable=reliable,
            depth_valid_fraction=valid_depth[0],depth_had_nonfinite=depth_nonfinite,
            crop_outside_fraction=((source_xy<0)|(source_xy>1)).any(-1).float().mean(),
            all_vertices_behind=(camera[:,2]<=.001).all(),render_calls=1,encoded_images_requested=1)
        return f,diag


def stack_current(features):
    return {k:torch.stack([f[k] for f in features]) for k in features[0]}
