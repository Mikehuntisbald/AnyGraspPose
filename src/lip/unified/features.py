"""Single student crop/render and isolated full-CAD RGB-D teacher targets."""
from dataclasses import dataclass,replace
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from lip.geometry.crop import crop_matrix,crop_images
from lip.geometry.so3 import log
from lip.data.jepa_pairs import normalize_rgb
from lip.jepa.contracts import FramePacket
from lip.engine.stream_state import FrameMeta

# DexYCB aligned depth is uint16 millimeters. Its saturated endpoint is not a
# usable surface measurement (it previously overflowed Utonia's 16-bit grid).
DEPTH_SATURATION_M=65.5345


def usable_depth(depth):
    return torch.where(torch.isfinite(depth)&(depth>0)&(depth<DEPTH_SATURATION_M),depth,0.)


@dataclass(frozen=True)
class Scene:
    rgb: torch.Tensor
    depth: torch.Tensor
    render: dict
    affine: torch.Tensor
    k_crop: torch.Tensor
    bounds: torch.Tensor
    pose: torch.Tensor
    diameter: float
    center: torch.Tensor
    state: torch.Tensor
    timestamp: float
    stream: str
    size_wh: tuple
    cad: dict


@dataclass(frozen=True)
class Observation:
    packet: FramePacket
    mid: torch.Tensor
    last: torch.Tensor
    cad_mid: torch.Tensor
    cad_last: torch.Tensor
    cad_valid: torch.Tensor
    geo: torch.Tensor
    geo_position: torch.Tensor
    geo_valid: torch.Tensor
    geo_observed: torch.Tensor
    object_xyz: torch.Tensor
    depth_valid: torch.Tensor
    state: torch.Tensor
    base: torch.Tensor
    diameter: torch.Tensor
    center: torch.Tensor
    metadata: FrameMeta
    fp_observed: torch.Tensor|None=None
    fp_pair: torch.Tensor|None=None
    dense_relation: torch.Tensor|None=None
    cad_relation: torch.Tensor|None=None
    geometry_image: torch.Tensor|None=None
    cad_surface_features: torch.Tensor|None=None
    cad_surface_geometry: torch.Tensor|None=None
    cad_surface_valid: torch.Tensor|None=None
    crop_rays: torch.Tensor|None=None
    rope_depth_stats: torch.Tensor|None=None
    measured_depth_m: torch.Tensor|None=None
    cad_atlas: tuple|None=None


@dataclass(frozen=True)
class TeacherTargets:
    real_mid: torch.Tensor|None
    real_last: torch.Tensor|None
    proxy_mid: torch.Tensor|None
    proxy_last: torch.Tensor|None
    visible_weight: torch.Tensor
    hidden_real_weight: torch.Tensor
    proxy_weight: torch.Tensor
    proxy_visible_weight: torch.Tensor
    proxy_hidden_weight: torch.Tensor
    visible_label: torch.Tensor
    support_label: torch.Tensor
    proxy_rgb: torch.Tensor|None
    real_rgb: torch.Tensor|None
    surface_xyz: torch.Tensor
    surface_depth_residual: torch.Tensor
    surface_depth_m: torch.Tensor
    geometry_weight: torch.Tensor
    geometry_visible_weight: torch.Tensor
    geometry_hidden_weight: torch.Tensor
    geometry_valid_label: torch.Tensor
    geometry_real_weight: torch.Tensor
    geometry_proxy_weight: torch.Tensor
    # Loss/scoring metadata only. Never part of Observation or student forward.
    camera_rotation: torch.Tensor|None=None
    camera_translation_d: torch.Tensor|None=None
    camera_rays: torch.Tensor|None=None
    base_depth_d: torch.Tensor|None=None
    cad_surface_target_real: torch.Tensor|None=None
    cad_surface_target_proxy: torch.Tensor|None=None
    cad_surface_weight_real: torch.Tensor|None=None
    cad_surface_weight_proxy: torch.Tensor|None=None
    cad_geometry_xyz: torch.Tensor|None=None
    cad_geometry_depth_m: torch.Tensor|None=None
    cad_geometry_valid: torch.Tensor|None=None
    real_geometry_eligible: torch.Tensor|None=None


@torch.no_grad()
@torch.autocast("cuda",enabled=False)
def prepare_scene(rgb,depth,base,mesh,k,timestamp,stream,cad,renderer,previous=None,last_timestamp=None,fast=False):
    device=base.device;d=float(mesh['diameter']);base=base.float();k=k.float()
    matrix_fn,image_fn=crop_matrix,crop_images
    if fast:
        from .execution_speed import crop_matrix_fast,crop_images_fast
        matrix_fn,image_fn=crop_matrix_fast,crop_images_fast
    affine,k_crop=matrix_fn(torch.as_tensor(mesh['vertices'],device=device),base,k,224,2.)
    crop=image_fn(rgb[None].float(),affine)
    observed=image_fn(depth[None].float(),affine,mode='nearest')
    observed=usable_depth(observed)
    bounds=image_fn(torch.ones_like(rgb[None,:1]),affine)>=.999
    rendered=renderer(cad['appearance'],base,k_crop,224)
    motion=base.new_zeros(3);rot=motion.clone();dt=base.new_zeros(1);mv=dt.clone()
    if previous is not None and last_timestamp is not None:
        rot=log(base[:3,:3]@previous[:3,:3].T);motion=(base[:3,3]-previous[:3,3])/d
        dt.fill_(1/30);mv.fill_(1)
    elapsed=0. if last_timestamp is None else timestamp-last_timestamp
    state=torch.cat((base[:3,:2].T.flatten(),base[:3,3]/d,base.new_tensor([d]).log(),
        torch.stack((k_crop[0,0],k_crop[1,1],k_crop[0,2],k_crop[1,2]))/224,
        rot,motion,dt,base.new_tensor([elapsed]),mv,(observed>0).float().mean().reshape(1)))
    assert state.shape==(24,)
    return Scene(crop,observed,rendered,affine,k_crop,bounds,base,d,
        torch.as_tensor(mesh['center'],device=device),state,float(timestamp),stream,(rgb.shape[-1],rgb.shape[-2]),cad)


def camera_points(depth,k):
    y,x=torch.meshgrid(torch.arange(224,device=depth.device,dtype=torch.float32),
                       torch.arange(224,device=depth.device,dtype=torch.float32),indexing='ij')
    rays=torch.stack((x,y,torch.ones_like(x)),-1)@torch.linalg.inv(k).T
    return rays*depth[0,0,:,:,None]


@torch.no_grad()
def observation_cloud(scene,mask,occlusion=None,max_points=4096):
    if occlusion is None:
        if mask.any():raise ValueError('Nonempty synthetic masks require a textured RGB-D occluder')
        depth=scene.depth;rgb=scene.rgb
    else:
        if not torch.equal(mask,occlusion.mask):raise ValueError('RGB-D occluder/mask mismatch')
        depth=occlusion.depth;rgb=occlusion.rgb
    camera=camera_points(depth,scene.k_crop)
    valid=(depth[0,0]>0)&scene.bounds[0,0]
    normal=torch.zeros_like(camera)
    normal[1:-1,1:-1]=torch.linalg.cross(camera[1:-1,2:]-camera[1:-1,:-2],camera[2:,1:-1]-camera[:-2,1:-1])
    normal=F.normalize(normal,dim=-1)
    normal=torch.where((normal*camera).sum(-1,keepdim=True)>0,-normal,normal)
    neighbor=F.avg_pool2d(valid[None,None].float(),3,1,1)[0,0]>=.999
    normal=normal*neighbor[...,None]
    object_xyz=(camera-scene.pose[:3,3])@scene.pose[:3,:3]/scene.diameter
    flat=valid.flatten().nonzero().flatten()
    if len(flat)>max_points:flat=flat[torch.linspace(0,len(flat)-1,max_points,device=flat.device).long()]
    cloud=dict(coord=object_xyz.reshape(-1,3)[flat],color=rgb[0].permute(1,2,0).reshape(-1,3)[flat],
               normal=(normal@scene.pose[:3,:3]).reshape(-1,3)[flat])
    return cloud,flat,rgb,object_xyz,valid,depth


@torch.autocast("cuda",enabled=False)
def encode_scenes(model,scenes,masks=None,cad_enabled=None,frame_id=0,occlusions=None):
    if getattr(model,'online_geometry',None)=='joint_geometry_image':
        from .two_stream import encode_two_stream
        return encode_two_stream(model,scenes,masks,cad_enabled,frame_id,occlusions)
    if getattr(model,'online_geometry',None)=='foundationpose':
        from .fp_features import encode_fp_scenes
        return encode_fp_scenes(model,scenes,masks,cad_enabled,frame_id,occlusions)
    b=len(scenes);device=scenes[0].rgb.device
    if occlusions is not None:
        if masks is not None:raise ValueError('Pass textured occlusions or empty masks, not both')
        masks=[o.mask for o in occlusions]
    masks=[torch.zeros_like(s.bounds) for s in scenes] if masks is None else masks
    occlusions=[None]*b if occlusions is None else occlusions
    cad_enabled=torch.ones(b,device=device,dtype=torch.bool) if cad_enabled is None else cad_enabled
    data=[observation_cloud(s,m,o,max_points=getattr(model,'observation_points',4096)) for s,m,o in zip(scenes,masks,occlusions)]
    rgb=torch.cat([x[2] for x in data]);render_rgb=torch.stack([s.render['rgb'] for s in scenes])
    timing=getattr(model,'profile_timing',None)
    def measure(name):return timing.record(name) if timing is not None else nullcontext()
    with measure('dino_observation_and_reference'),torch.autocast(device.type,dtype=torch.bfloat16):
        mid,last=model.encoder(normalize_rgb(torch.cat((rgb,render_rgb))))
    with measure('utonia_observation'):encoded=model.utonia([x[0] for x in data])
    geometry=[];positions=[];valids=[];xyz_patches=[];depth_patches=[]
    for lane,(s,(cloud,pixels,_,xyz,valid,observed_depth),feat) in enumerate(zip(scenes,data,encoded)):
        geo=torch.zeros(256,model.utonia.feature_dim,device=device)
        gp=torch.zeros(256,6,device=device);counts=torch.zeros(256,device=device)
        if feat is not None:
            y,x=pixels//224,pixels%224;cell=(y//14)*16+x//14
            order=cell.argsort(stable=True)
            counts=torch.bincount(cell,minlength=256).float()
            ptr=F.pad(counts.long().cumsum(0),(1,0))
            from torch_scatter import segment_csr
            geo=segment_csr(feat[order],ptr,reduce="sum")
            residual=(observed_depth[0,0].flatten()[pixels]-s.render['depth'][0].flatten()[pixels])/s.diameter
            residual=residual*cad_enabled[lane]
            coord=torch.cat((cloud['coord'],torch.stack((x,y),-1).float()/112-1,residual[:,None]),-1)
            gp=segment_csr(coord[order],ptr,reduce='sum')
        geo/=counts.clamp_min(1)[:,None];gp/=counts.clamp_min(1)[:,None]
        xyz_patches.append(gp[:,:3]);depth_patches.append(counts>0)
        indices=torch.linspace(0,len(s.cad['coord'])-1,128,device=device).long()
        cad_xyz=s.cad['coord'][indices]
        camera=(cad_xyz*s.diameter)@s.pose[:3,:3].T+s.pose[:3,3]
        uv=camera@s.k_crop.T;uv=uv[:,:2]/uv[:,2:].clamp_min(.001)
        cp=torch.cat((cad_xyz,uv/112-1,torch.zeros(128,1,device=device)),-1)
        geometry.append(torch.cat((geo,s.cad['features'][indices])))
        positions.append(torch.cat((gp,cp)))
        valids.append(torch.cat((counts>0,(camera[:,2]>.001)&cad_enabled[lane])))
    packet=FramePacket(normalize_rgb(rgb),torch.cat([s.bounds for s in scenes]),torch.stack([s.affine for s in scenes]),
        torch.tensor([s.timestamp for s in scenes],device=device,dtype=torch.float64),tuple(s.stream for s in scenes),
        rgb.new_tensor([[32.,32.,192.,192.]]).expand(b,-1),rgb.new_tensor([s.size_wh for s in scenes]))
    valid=F.avg_pool2d(packet.pixel_valid.float(),14,14).flatten(1)>=.999
    metadata=FrameMeta(packet.timestamp_s,torch.full((b,),frame_id,device=device,dtype=torch.long),
        torch.arange(b,device=device),torch.cat((F.adaptive_max_pool2d(packet.pixel_valid.float(),4).flatten(1)>0,
        valid.any(-1,keepdim=True)),1),torch.zeros(b,17,device=device))
    observed_state=torch.stack([s.state for s in scenes]).clone()
    observed_state[:,-1]=torch.stack([row[4].float().mean() for row in data])
    return Observation(packet,mid[:b],last[:b],mid[b:],last[b:],
        valid&cad_enabled[:,None],torch.stack(geometry),torch.stack(positions),torch.stack(valids),
        torch.arange(384,device=device)[None].expand(b,-1)<256,torch.stack(xyz_patches),torch.stack(depth_patches),
        observed_state,torch.stack([s.pose for s in scenes]),
        rgb.new_tensor([s.diameter for s in scenes]),torch.stack([s.center for s in scenes]),metadata)


def valid_real_geometry(depth, canonical_xyz, max_radius_d=None):
    valid=(depth>0)&torch.isfinite(depth)
    rejected=torch.zeros_like(valid)
    if max_radius_d is not None:
        if max_radius_d<=0:raise ValueError('Positive physical target radius required')
        plausible=torch.isfinite(canonical_xyz).all(1,keepdim=True)&(canonical_xyz.norm(dim=1,keepdim=True)<=max_radius_d)
        rejected=valid&~plausible
        valid=valid&plausible
    return valid,rejected


@torch.no_grad()
@torch.autocast("cuda",enabled=False)
def build_teachers(encoder,scenes,gt_poses,visible_masks,added_masks,renderer,encoded_targets=None,real_features=None,reuse_real=None,real_geometry_max_radius_d=None,fast=False,batch_render=False,vectorized=False,geometry_only=False):
    if geometry_only and not vectorized:raise ValueError('Geometry-only teacher requires vectorized targets')
    if vectorized:
        if encoded_targets is not None or real_features is not None:raise ValueError('Vector teacher expects fresh EMA targets')
        from .fast_teacher import build_fast_teacher
        return build_fast_teacher(encoder,scenes,gt_poses,visible_masks,added_masks,renderer,real_geometry_max_radius_d,batch_render,geometry_only=geometry_only)
    image_fn,point_fn=crop_images,camera_points
    if fast:
        from .execution_speed import crop_images_fast,camera_points_fast
        image_fn,point_fn=crop_images_fast,camera_points_fast
    renders=renderer.render_many([s.cad['appearance'] for s in scenes],gt_poses,[s.k_crop for s in scenes],224) if batch_render else None
    proxies=[];visible=[];support=[];interior=[];xyz=[];depth=[];residual=[];geometry=[];geo_visible=[];geo_hidden=[];geo_valid=[]
    geometry_real=[];geometry_proxy=[];proxy_eligible=[];real_eligible=[]
    cad_geometry_xyz=[];cad_geometry_depth=[];cad_geometry_valid=[]
    for lane,(s,pose,mask,added) in enumerate(zip(scenes,gt_poses,visible_masks,added_masks)):
        render=renders[lane] if renders is not None else renderer(s.cad['appearance'],pose.float(),s.k_crop,224)
        v=image_fn(mask[None].float(),s.affine,mode='nearest')>.5
        silhouette=render['mask'][None,None]
        cad_geometry_xyz.append(render['xyz'][None]/s.diameter)
        cad_geometry_depth.append(render['depth'][None])
        cad_geometry_valid.append(silhouette&s.bounds&(render['depth'][None]>0))
        hidden=silhouette&~v
        # The entire GT silhouette uses one consistent CAD appearance/depth source.
        proxy=torch.where(silhouette,render['rgb'][None],s.rgb)
        proxies.append(proxy);visible.append(v);support.append(silhouette)
        # Exclude two-pixel boundaries and annotation/render disagreement.
        def erode(m):return -F.max_pool2d(-m.float(),5,1,2)>.999
        good=erode(silhouette)&s.bounds&(render['depth'][None]>0)
        known=erode(v)&silhouette&s.bounds
        hidden_good=erode(hidden)&s.bounds
        artificial=v&added
        artificial_interior=erode(artificial)&known
        # A measured target takes precedence wherever the original object was
        # visible and then synthetically hidden. Do not replace missing real
        # depth with a CAD measurement in that region.
        real_xyz=((point_fn(s.depth,s.k_crop)-pose[:3,3])@pose[:3,:3]/s.diameter).permute(2,0,1)[None]
        real_valid,rejected_depth=valid_real_geometry(s.depth,real_xyz,real_geometry_max_radius_d)
        real_good=artificial_interior&erode(real_valid)
        # The full-CAD image defines the feature target's context, not the loss
        # domain. Only naturally hidden object pixels receive CAD supervision.
        proxy_good=good&hidden_good
        target_xyz=torch.where(artificial.expand_as(real_xyz),real_xyz,render['xyz'][None]/s.diameter)
        target_depth=torch.where(artificial,s.depth,render['depth'][None])
        xyz.append(target_xyz);depth.append(target_depth)
        residual.append((target_depth-s.pose[2,3])/s.diameter)
        geometry.append(real_good|proxy_good)
        geo_visible.append(real_good);geo_hidden.append(proxy_good)
        geometry_real.append(real_good);geometry_proxy.append(proxy_good)
        proxy_eligible.append(hidden_good)
        real_eligible.append(artificial_interior)
        interior.append((known,good,hidden_good))
        label=(silhouette&(render['depth'][None]>0)).float()
        label=torch.where(artificial,real_valid.float(),label)
        # A corrupt measurement is unknown, not evidence of absent surface.
        label=label.masked_fill(artificial&rejected_depth,float('nan'))
        label=label.masked_fill(~s.bounds,float('nan'))
        label=label.masked_fill(v&~silhouette,float('nan'))
        label=label.masked_fill(v&~added,float('nan'))
        # Do not train geometry-validity decisions at uncertain silhouette edges.
        boundary=~erode(silhouette)&~erode(~silhouette)
        geo_valid.append(label.masked_fill(boundary,float('nan')))
    original=torch.cat([s.rgb for s in scenes]);proxy=torch.cat(proxies);b=len(scenes)
    if encoded_targets is None:
        if real_features is None:
            with torch.autocast(original.device.type,dtype=torch.bfloat16):
                mid,last=encoder(normalize_rgb(torch.cat((original,proxy))))
            real_mid,real_last,proxy_mid,proxy_last=mid[:b].float(),last[:b].float(),mid[b:].float(),last[b:].float()
        else:
            if reuse_real is None or len(reuse_real)!=b:raise ValueError('Explicit unchanged-RGB lanes required')
            needed=[i for i,unchanged in enumerate(reuse_real) if not unchanged]
            with torch.autocast(original.device.type,dtype=torch.bfloat16):
                mid,last=encoder(normalize_rgb(torch.cat((original[needed],proxy))))
            real_mid,real_last=[x.detach().float().clone() for x in real_features]
            if needed:real_mid[needed]=mid[:len(needed)].float();real_last[needed]=last[:len(needed)].float()
            proxy_mid,proxy_last=mid[len(needed):].float(),last[len(needed):].float()
    else:real_mid,real_last,proxy_mid,proxy_last=encoded_targets
    pool=lambda x:F.avg_pool2d(x.float(),14,14).flatten(1)
    bounds=pool(torch.cat([s.bounds for s in scenes]))>=.999
    known=pool(torch.cat([v[0] for v in interior]))>=.9
    complete_proxy=pool(torch.cat(proxy_eligible))>=.9
    hidden_proxy=pool(torch.cat([v[2] for v in interior]))>=.9
    real_hidden=pool(torch.cat(real_eligible))>=.9
    added=pool(torch.cat(added_masks))
    visible_label=pool(torch.cat(visible)&~torch.cat(added_masks)).masked_fill(~bounds,float('nan'))
    support_label=pool(torch.cat(support)).masked_fill(~bounds,float('nan'))
    return TeacherTargets(real_mid,real_last,proxy_mid,proxy_last,
        known*bounds*(1-added),real_hidden*bounds,complete_proxy*bounds,
        complete_proxy*known*bounds,complete_proxy*hidden_proxy*bounds,
        visible_label,support_label,proxy,original,torch.cat(xyz),torch.cat(residual),torch.cat(depth),
        torch.cat(geometry),torch.cat(geo_visible),torch.cat(geo_hidden),torch.cat(geo_valid),torch.cat(geometry_real),torch.cat(geometry_proxy),
        torch.stack([p[:3,:3].float() for p in gt_poses]),
        torch.stack([p[:3,3].float()/s.diameter for p,s in zip(gt_poses,scenes)]),
        torch.stack([point_fn(torch.ones_like(s.depth),s.k_crop).permute(2,0,1) for s in scenes]),
        torch.stack([s.pose[2,3]/s.diameter for s in scenes]),
        cad_geometry_xyz=torch.cat(cad_geometry_xyz),cad_geometry_depth_m=torch.cat(cad_geometry_depth),cad_geometry_valid=torch.cat(cad_geometry_valid))
