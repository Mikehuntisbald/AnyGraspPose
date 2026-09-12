"""Transactional online API. No labels, FP, critic, or implicit reinitialization."""
from dataclasses import replace
import hashlib
import math
import torch
from lip.engine.stream_state import FrameMeta,TemporalCache,RingCache,CrossCache,StreamState,SourceGeometry
from lip.engine.stream_features import build_current_features,stack_current,mesh_to_device
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import center_pose,original_pose


def initialize(model,T0_original,mesh,K,stream_id,timestamp0,*,object_id='object',camera_id='camera',
               mesh_hash=None,image_shape=(480,640),generation=0):
    if T0_original is None:raise ValueError('An explicit known initial pose is required')
    if not stream_id or not math.isfinite(float(timestamp0)):raise ValueError('Invalid stream or timestamp')
    device=next(model.parameters()).device
    pose=torch.as_tensor(T0_original,dtype=torch.float32,device=device).detach().clone()
    k=torch.as_tensor(K,dtype=torch.float32,device=device).detach().clone()
    if pose.shape!=(4,4) or k.shape!=(3,3):raise ValueError('Expected 4x4 initial pose and 3x3 K')
    if not torch.isfinite(pose).all() or not torch.isfinite(k).all():raise ValueError('Nonfinite initialization')
    eye=torch.eye(3,device=device)
    if not torch.allclose(pose[:3,:3].T@pose[:3,:3],eye,atol=2e-3) or pose[2,3]<=0 or not torch.allclose(pose[3],pose.new_tensor([0,0,0,1])):
        raise ValueError('Initial pose must be a valid object-to-camera transform in meters')
    if k[0,0]<=0 or k[1,1]<=0:raise ValueError('Invalid focal length')
    if mesh_hash is None:
        digest=hashlib.sha256()
        for key in ('vertices','faces','center','diameter'):
            digest.update(torch.as_tensor(mesh[key]).cpu().numpy().tobytes())
        mesh_hash=digest.hexdigest()
    resident=mesh_to_device(mesh,device)
    pose=center_pose(pose,resident['center'].float())
    cache=RingCache.empty(model.memory_frames) if model.cache_kind=='ring' else TemporalCache(capacity=model.memory_frames)
    if model.architecture_id=='stream_dual_cross':cache=CrossCache(cache)
    return StreamState(cache,pose,float(timestamp0),None,None,0,str(stream_id),str(object_id),str(camera_id),
        str(mesh_hash),resident,k,tuple(image_shape),model.weights_version,model.parameter_versions(),generation=generation,cache_contract=model.cache_contract)


def failure(state,status,needs_reinit=True,diagnostics=None):
    return dict(status=status,needs_reinit=needs_reinit,pose_centered=state.pose_centered,
        pose_original=original_pose(state.pose_centered,state.mesh['center'].float()),
        delta_rotvec=None,delta_center_norm=None,latent=None,confidence=None,diagnostics=diagnostics or {}),state


def step(model,rgb,depth,timestamp,state,*,renderer=None,precision='fp32',image_size=224,crop_expansion=2.,
         stream_id=None,object_id=None,camera_id=None,mesh_hash=None,K=None,profiler=None):
    if not isinstance(state,StreamState):raise TypeError('Expected independent StreamState')
    if state.cache_contract!=model.cache_contract:return failure(state,'cache_contract_changed')
    if state.pending_pose is not None:return failure(state,'uncommitted_proposal')
    if state.weights_version!=model.weights_version or state.parameter_versions!=model.parameter_versions():
        return failure(state,'weights_changed')
    for name,value in [('stream_id',stream_id),('object_id',object_id),('camera_id',camera_id),('mesh_hash',mesh_hash)]:
        if value is not None and str(value)!=getattr(state,name):return failure(state,'identity_changed')
    if tuple(rgb.shape[-2:])!=state.image_shape or tuple(depth.shape[-2:])!=state.image_shape:
        return failure(state,'image_shape_changed')
    if K is not None and not torch.equal(torch.as_tensor(K,device=state.K.device,dtype=state.K.dtype),state.K):
        return failure(state,'intrinsics_changed')
    now=float(timestamp)
    if not math.isfinite(now) or now<=state.timestamp:return failure(state,'timestamp_not_increasing')
    if now-state.timestamp>model.max_gap_seconds:return failure(state,'time_gap')
    if precision not in ('bf16','fp32'):raise ValueError('Explicit fp32 or bf16 required')
    if precision=='bf16' and state.K.device.type!='cuda':raise ValueError('BF16 online path requires CUDA')
    renderer=renderer or Renderer(state.K.device)
    rgb=torch.as_tensor(rgb,device=state.K.device)
    depth=torch.as_tensor(depth,device=state.K.device)
    if rgb.dtype==torch.uint8:rgb=rgb.float()/255
    try:
        features,diag=build_current_features(rgb,depth,state.pose_centered,state.K,state.mesh,renderer,
            now,state.timestamp,state.previous_pose,state.previous_timestamp,image_size,crop_expansion)
    except ValueError as exc:return failure(state,'invalid_input',diagnostics={'error':str(exc)})
    device=state.K.device;frame_id=state.frame_id+1
    meta=FrameMeta(torch.tensor([now],dtype=torch.float64,device=device),torch.tensor([frame_id],device=device),
        torch.tensor([state.generation],device=device),torch.ones(1,17,device=device,dtype=torch.bool),diag['role_bias'][None])
    cache=state.cache.functional() if isinstance(state.cache,RingCache) else state.cache
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
        proposal,next_cache=model(stack_current([features]),meta,cache,profiler)
    checks=[torch.isfinite(t).all() for t in proposal.values() if isinstance(t,torch.Tensor)]
    checks += [torch.isfinite(t).all() for layer in next_cache.layers for b in layer[-1:] for t in (b.key,b.value)]
    if isinstance(next_cache,CrossCache):
        checks += [torch.isfinite(t).all() for b in next_cache.contexts[-1:] for t in (b.key,b.value)]
    finite=bool(torch.stack(checks).all()) # One host decision, not one synchronization per layer.
    if not finite:return failure(state,'nonfinite_proposal',diagnostics=diag)
    proposal={k:(v[0] if isinstance(v,torch.Tensor) else v) for k,v in proposal.items()}
    proposal.update(status='ok',needs_reinit=False,diagnostics=diag,frame_id=frame_id,timestamp=now,
                    stream_id=state.stream_id,weights_version=model.weights_version)
    if isinstance(state.cache,RingCache):
        next_cache=state.cache.append([l[-1] for l in next_cache.layers],meta)
    source=SourceGeometry(now,frame_id,state.stream_id,state.object_id,state.camera_id,
        diag['A'].detach().clone(),diag['K_crop'].detach().clone(),state.pose_centered.detach().clone(),state.image_shape,
        features['object_diameter_m'].detach().clone(),state.mesh_hash,meta.key_valid.detach().clone(),diag['silhouette_token_fraction'].detach().clone())
    next_state=replace(state,cache=next_cache,pending_pose=proposal['pose_centered'],pending_timestamp=now,
        frame_id=frame_id,source_metadata=(state.source_metadata+(source,))[-model.memory_frames:])
    return proposal,next_state


def commit(model,proposal,next_state):
    if proposal.get('status')!='ok' or proposal.get('needs_reinit'):raise ValueError('Cannot commit failed proposal')
    if next_state.pending_pose is None or proposal['pose_centered'] is not next_state.pending_pose:
        raise ValueError('Proposal does not belong to this pending state')
    if model.weights_version!=next_state.weights_version or model.parameter_versions()!=next_state.parameter_versions:
        raise ValueError('Weights changed before commit; reset required')
    if not torch.isfinite(next_state.pending_pose).all():raise ValueError('Nonfinite pose at commit')
    return replace(next_state,previous_pose=next_state.pose_centered.detach(),previous_timestamp=next_state.timestamp,
        pose_centered=next_state.pending_pose.detach(),timestamp=next_state.pending_timestamp,pending_pose=None,pending_timestamp=None)


def correct(model,state,pose_original,timestamp=None,relocalization=False):
    if state.pending_pose is not None:raise ValueError('Commit or discard pending proposal before external correction')
    when=state.timestamp if timestamp is None else float(timestamp)
    checked=initialize(model,pose_original,state.mesh,state.K,state.stream_id,when,
        object_id=state.object_id,camera_id=state.camera_id,mesh_hash=state.mesh_hash,image_shape=state.image_shape,generation=state.generation+1)
    if relocalization:return checked
    if when!=state.timestamp:raise ValueError('Small correction must refer to the last accepted timestamp')
    # Preserve the immutable source/crop/KV history; only future input uses this pose.
    return replace(state,pose_centered=checked.pose_centered)
