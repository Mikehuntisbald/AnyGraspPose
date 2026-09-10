import torch
from lip.geometry.crop import crop_matrix, crop_images, geometry_channels


@torch.no_grad()
def build_features(rgb, depth, history, times, frame_valid, pose_valid, base, k, mesh,
                   renderer, size=224, expansion=2.):
    """No GT argument: history contains accepted estimates only; current is unavailable."""
    device=base.device
    if bool(pose_valid[-1]):raise ValueError('Current-frame pose must be unavailable to the model')
    with torch.autocast(device.type, enabled=False):
        base=base.float(); k=k.float()
        vertices=torch.as_tensor(mesh['vertices'], device=device)
        d=torch.as_tensor(mesh['diameter'], device=device)
        a,kc=crop_matrix(vertices, base, k, size, expansion)
        rgb=crop_images(rgb.float(), a, size)
        depth=crop_images(depth.float(), a, size, 'nearest')
        rd,xyz=renderer(mesh, base, kc, size)
        geom=geometry_channels(depth, rd[None], xyz[None], d, base[2, 3])
        rr=history[:, :3, :3] @ base[:3, :3].T
        hs=torch.cat((rr[:, :, :2].transpose(-1, -2).flatten(1), (history[:, :3, 3]-base[:3, 3])/d,
                      pose_valid[:, None].float()), -1)
        hs=hs*pose_valid[:, None]
        bs=torch.cat((base[:3, :2].T.flatten(), base[:3, 3]/d, d.log()[None],
                      torch.stack((kc[0, 0], kc[1, 1], kc[0, 2], kc[1, 2]))/size))
        mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None]
        std=rgb.new_tensor([.229,.224,.225])[None,:,None,None]
        return dict(rgb=(rgb-mean)/std, geometry=geom, history_state=hs, base_state=bs,
                    time_offsets_sec=times-times[-1], frame_valid=frame_valid, history_pose_valid=pose_valid,
                    T_base_centered=base, object_diameter_m=d,
                    mesh_center=torch.as_tensor(mesh['center'], device=device)), dict(A=a, K_crop=kc, render_depth=rd, crop_rgb=rgb)


def stack_features(features):
    return {k:torch.stack([f[k] for f in features]) for k in features[0]}
