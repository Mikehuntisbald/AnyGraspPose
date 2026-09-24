"""Dense FP observation path; immutable full-CAD Utonia tokens stay separate."""
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from lip.data.jepa_pairs import normalize_rgb
from lip.jepa.contracts import FramePacket
from lip.engine.stream_state import FrameMeta
from .features import Observation, camera_points, usable_depth


@torch.no_grad()
@torch.autocast('cuda', enabled=False)
def encode_fp_scenes(model, scenes, masks=None, cad_enabled=None, frame_id=0, occlusions=None):
    b = len(scenes); device = scenes[0].rgb.device
    if masks is not None and occlusions is not None:
        raise ValueError('Pass textured RGB-D occlusions or empty masks')
    if masks is not None and any(bool(mask.any()) for mask in masks):
        raise ValueError('Synthetic masks need real RGB-D occluder textures')
    occlusions = [None] * b if occlusions is None else occlusions
    cad_enabled = torch.ones(b, device=device, dtype=torch.bool) if cad_enabled is None else cad_enabled
    rgb = torch.cat([s.rgb if o is None else o.rgb for s, o in zip(scenes, occlusions)])
    depth = usable_depth(torch.cat([s.depth if o is None else o.depth for s, o in zip(scenes, occlusions)]))
    bounds = torch.cat([s.bounds for s in scenes]); valid = (depth > 0) & bounds
    base = torch.stack([s.pose for s in scenes]); diameter = rgb.new_tensor([s.diameter for s in scenes])
    camera = torch.stack([camera_points(depth[i:i+1], s.k_crop) for i, s in enumerate(scenes)])
    xyz = torch.einsum('bhwc,bcd->bhwd', camera-base[:, None, None, :3, 3], base[:, :3, :3]) / diameter[:, None, None, None]
    render_rgb = torch.stack([s.render['rgb'] for s in scenes])
    render_depth = torch.stack([s.render['depth'] for s in scenes])
    render_camera = torch.stack([s.render['xyz'].permute(1, 2, 0) @ s.pose[:3, :3].T + s.pose[:3, 3] for s in scenes])
    timing = getattr(model, 'profile_timing', None)
    measure = lambda name: timing.record(name) if timing is not None else nullcontext()
    with measure('dino_observation_and_reference'), torch.autocast(device.type, dtype=torch.bfloat16):
        mid, last = model.encoder(normalize_rgb(torch.cat((rgb, render_rgb))))
    with measure('fp_observation_and_reference'):
        if getattr(model, 'fp_observation_only', False):
            fp_observed = model.fp.encode_observation(rgb, camera.permute(0, 3, 1, 2), depth,
                                                     base[:, :3, 3], diameter)
            fp_pair = None
        else:
            fp_observed, fp_pair = model.fp(rgb, camera.permute(0, 3, 1, 2), depth,
                                          render_rgb, render_camera.permute(0, 3, 1, 2), render_depth, base[:, :3, 3], diameter)
    # Geometry and validity pool actual measurements. No fabricated depth for
    # holes or absent depth; the RGB/CAD/history branches remain available.
    pool = lambda x: F.avg_pool2d(x, 14, 14).flatten(2).transpose(1, 2)
    mass = pool(valid.float())
    object_xyz = pool(xyz.permute(0, 3, 1, 2)*valid) / mass.clamp_min(1e-6)
    y, x = torch.meshgrid(torch.arange(224, device=device), torch.arange(224, device=device), indexing='ij')
    xy = torch.stack((x, y), 0).float()[None]/112-1
    uv = pool(xy*valid)/mass.clamp_min(1e-6)
    residual = pool((depth-render_depth)/diameter[:, None, None, None]*valid)/mass.clamp_min(1e-6)
    observed_position = torch.cat((object_xyz, uv, residual*cad_enabled[:, None, None]), -1)
    coordinates=[]; features=[]; cad_positions=[]; cad_valid=[]
    for s in scenes:
        indices=torch.linspace(0,len(s.cad['coord'])-1,128,device=device).long()
        coord=s.cad['coord'][indices]; points=(coord*s.diameter)@s.pose[:3,:3].T+s.pose[:3,3]
        projected=points@s.k_crop.T; projected=projected[:,:2]/projected[:,2:].clamp_min(.001)
        cad_positions.append(torch.cat((coord,projected/112-1,torch.zeros(128,1,device=device)),-1))
        features.append(s.cad['features'][indices]);cad_valid.append(points[:,2]>.001)
    packet=FramePacket(normalize_rgb(rgb),bounds,torch.stack([s.affine for s in scenes]),
        torch.tensor([s.timestamp for s in scenes],device=device,dtype=torch.float64),tuple(s.stream for s in scenes),
        rgb.new_tensor([[32.,32.,192.,192.]]).expand(b,-1),rgb.new_tensor([s.size_wh for s in scenes]))
    pixel_valid=F.avg_pool2d(bounds.float(),14,14).flatten(1)>=.999
    metadata=FrameMeta(packet.timestamp_s,torch.full((b,),frame_id,device=device,dtype=torch.long),
        torch.arange(b,device=device),torch.cat((F.adaptive_max_pool2d(bounds.float(),4).flatten(1)>0,
        pixel_valid.any(-1,keepdim=True)),1),torch.zeros(b,17,device=device))
    state=torch.stack([s.state for s in scenes]).clone();state[:,-1]=valid.float().mean((1,2,3))
    dense_relation=cad_relation=None
    if getattr(model,'jepa_pose_geometry',False):
        from .dynamic_geometry import dynamic_relations
        dense_relation,cad_relation=dynamic_relations(scenes,camera,depth,render_camera,render_depth,
            bounds,base,diameter,cad_enabled)
    # geo contains only 128 static CAD rows in this architecture. Online
    # FP rows have their own 128/512-D fields and distinct trainable projections.
    return Observation(packet,mid[:b],last[:b],mid[b:],last[b:],pixel_valid&cad_enabled[:,None],
        torch.stack(features),torch.cat((observed_position,torch.stack(cad_positions)),1),
        torch.cat((mass[...,0]>0,torch.stack(cad_valid)&cad_enabled[:,None]),1),
        torch.arange(384,device=device)[None].expand(b,-1)<256,object_xyz,mass[...,0]>0,
        state,base,diameter,torch.stack([s.center for s in scenes]),metadata,fp_observed,fp_pair,dense_relation,cad_relation)
