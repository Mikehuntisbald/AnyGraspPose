"""Batch current-frame crops and state features; reuse the existing mesh renderer."""
import torch
from torch.nn import functional as F
from lip.geometry.crop import crop_matrix, geometry_channels
from lip.geometry.so3 import log


def crop_batch(images, inverse, size, mode='bilinear'):
    h, w = images.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(size, device=images.device, dtype=inverse.dtype),
                           torch.arange(size, device=images.device, dtype=inverse.dtype), indexing='ij')
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
    uv = pixels[None] @ inverse.transpose(-1, -2)
    grid = torch.stack((2*(uv[..., 0]+.5)/w-1, 2*(uv[..., 1]+.5)/h-1), -1)
    return F.grid_sample(images, grid.reshape(len(images), size, size, 2), mode=mode,
                         padding_mode='zeros', align_corners=False)


@torch.no_grad()
def build_current_batch(rgb, depth, base, k, meshes, renderer, timestamp,
                        last_accepted_timestamp, previous_pose=None, previous_timestamp=None,
                        size=224, expansion=2., beta=1., epsilon=.1):
    """Same current observation/accepted-pose contract as build_current_features."""
    b = len(base)
    if rgb.ndim != 4 or depth.ndim != 4 or rgb.shape[:2] != (b, 3) or depth.shape[:2] != (b, 1) or rgb.shape[-2:] != depth.shape[-2:]:
        raise ValueError('Expected current RGB B3HW and depth B1HW')
    device = base.device
    with torch.autocast(device.type, enabled=False):
        rgb, depth, base, k = (x.to(device).float() for x in (rgb, depth, base, k))
        diameter = torch.stack([torch.as_tensor(m['diameter'], device=device, dtype=torch.float32) for m in meshes])
        # One batch-level check preserves fail-fast behavior without per-lane syncs.
        valid = torch.stack((torch.isfinite(rgb).all(), torch.isfinite(base).all(),
                             torch.isfinite(k).all(), (torch.isfinite(diameter)&(diameter>0)).all())).all()
        if not bool(valid):
            raise ValueError('Nonfinite RGB / accepted pose / intrinsics or invalid diameter')
        nonfinite = (~torch.isfinite(depth)).flatten(1).any(1)
        depth = torch.where(torch.isfinite(depth)&(depth>0), depth, 0.)
        # Mesh topology varies across lanes. Preserve the established crop bounds
        # and rasterization implementation; batch all image/state work around it.
        matrices = [crop_matrix(torch.as_tensor(m['vertices'],device=device,dtype=torch.float32), base[i], k[i], size, expansion) for i,m in enumerate(meshes)]
        a = torch.stack([x[0] for x in matrices])
        kc = torch.stack([x[1] for x in matrices])
        inverse = torch.linalg.inv(a)
        crop_rgb = crop_batch(rgb, inverse, size)
        crop_depth = crop_batch(depth, inverse, size, 'nearest')
        renders = [renderer(m, base[i], kc[i], size) for i,m in enumerate(meshes)]
        rendered = torch.stack([r[0] for r in renders])
        xyz = torch.stack([r[1] for r in renders])
        geom = geometry_channels(crop_depth, rendered, xyz, diameter[:,None,None,None], base[:,2,3,None,None,None])
        fraction = F.adaptive_avg_pool2d((rendered>0).float(), 4)[:,0].flatten(1)
        reliable = (rendered>0).flatten(1).any(1)
        role = torch.where(reliable[:,None], beta*torch.log(epsilon+(1-epsilon)*fraction), torch.zeros_like(fraction))
        role = torch.cat((role, role.new_zeros(b,1)), 1)
        cell = (torch.arange(4, device=device, dtype=torch.float32)+.5)*size/4-.5
        yy, xx = torch.meshgrid(cell,cell,indexing='ij')
        pixels = torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(16,3)
        uv = pixels[None] @ inverse.transpose(-1,-2)
        h,w = rgb.shape[-2:]
        source_xy = uv[:,:,:2]/uv.new_tensor([w,h])
        now = torch.as_tensor(timestamp,device=device,dtype=torch.float64)
        last = torch.as_tensor(last_accepted_timestamp,device=device,dtype=torch.float64)
        delta = base.new_zeros(b,3);motion = base.new_zeros(b,3)
        dt = base.new_zeros(b,1);motion_valid = base.new_zeros(b,1)
        if previous_pose is not None and previous_timestamp is not None:
            past = torch.as_tensor(previous_timestamp,device=device,dtype=torch.float64)
            valid_motion = (last>past)[:,None]
            previous_pose = previous_pose.float()
            delta = torch.where(valid_motion,log(base[:,:3,:3]@previous_pose[:,:3,:3].transpose(-1,-2)),delta)
            motion = torch.where(valid_motion,(base[:,:3,3]-previous_pose[:,:3,3])/diameter[:,None],motion)
            dt = torch.where(valid_motion,(last-past).float()[:,None],dt)
            motion_valid = valid_motion.float()
        valid_depth = (crop_depth>0).float().flatten(1).mean(1,keepdim=True)
        state = torch.cat((base[:,:3,:2].transpose(-1,-2).flatten(1),base[:,:3,3]/diameter[:,None],
                           diameter.log()[:,None],torch.stack((kc[:,0,0],kc[:,1,1],kc[:,0,2],kc[:,1,2]),1)/size,
                           delta,motion,dt,(now-last).float()[:,None],motion_valid,valid_depth),1)
        mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None]
        std=rgb.new_tensor([.229,.224,.225])[None,:,None,None]
        features=dict(rgb=(crop_rgb-mean)/std,geometry=geom,state_input=state,source_xy=source_xy,
                      T_base_centered=base,object_diameter_m=diameter,
                      mesh_center=torch.stack([torch.as_tensor(m['center'],device=device,dtype=torch.float32) for m in meshes]))
        diag=dict(A=a,K_crop=kc,T_base_source=base.clone(),silhouette_token_fraction=fraction,
                  role_bias=role,geometry_reliable=reliable,depth_valid_fraction=valid_depth[:,0],
                  depth_had_nonfinite=nonfinite,render_calls=b,encoded_images_requested=b)
        return features,diag
