"""Causal native runtime with no teacher/label interface."""
from dataclasses import dataclass,replace
import math
import torch
from lip.geometry.so3 import center_pose,original_pose
from lip.engine.stream_runtime import valid_pose
from lip.unified.features import prepare_scene,encode_scenes


@dataclass(frozen=True)
class State:
    pose_centered: torch.Tensor
    previous_pose: torch.Tensor|None
    timestamp: float
    frame_id: int
    stream_id: str
    memory: object
    mesh: dict
    K: torch.Tensor
    cad: dict
    versions: tuple
    history_enabled: bool
    image_shape: tuple


@torch.no_grad()
def initialize(model,rgb,depth,pose_original,mesh,k,timestamp,stream,renderer,cad,*,enabled=True,precision='bf16'):
    device=rgb.device;pose=torch.as_tensor(pose_original,device=device,dtype=torch.float32)
    pose=center_pose(pose,torch.as_tensor(mesh['center'],device=device))
    if not valid_pose(pose):raise ValueError('Invalid native initializer')
    scene=prepare_scene(rgb,depth,pose,mesh,k,timestamp,stream,cad,renderer)
    obs=encode_scenes(model,[scene])
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
        _,memory=model(obs,history_enabled=torch.tensor([enabled],device=device))
    return State(pose,None,float(timestamp),0,stream,memory.detach(),mesh,k,cad,
        tuple(p._version for p in model.parameters()),enabled,tuple(rgb.shape[-2:]))


@torch.no_grad()
def step(model,rgb,depth,timestamp,state,renderer,*,precision='bf16',stream_id=None):
    if state.versions!=tuple(p._version for p in model.parameters()):raise ValueError('Weights changed; reset required')
    if stream_id is not None and stream_id!=state.stream_id:raise ValueError('Cross-stream call')
    if not math.isfinite(timestamp) or not state.timestamp<timestamp<=state.timestamp+.5:raise ValueError('Noncausal or excessive time gap')
    if tuple(rgb.shape[-2:])!=state.image_shape:raise ValueError('Image shape changed')
    scene=prepare_scene(rgb,depth,state.pose_centered,state.mesh,state.K,timestamp,state.stream_id,
        state.cad,renderer,state.previous_pose,state.timestamp)
    obs=encode_scenes(model,[scene],frame_id=state.frame_id+1)
    with torch.autocast(rgb.device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
        result,memory=model(obs,state.memory,torch.tensor([state.history_enabled],device=rgb.device))
    pose=result['pose_centered'][0];status='ok'
    if not valid_pose(pose):pose=state.pose_centered;status='invalid_pose'
    if not all(torch.isfinite(r.features).all() for r in memory.objects+memory.contexts):
        memory=state.memory;status='nonfinite_memory';pose=state.pose_centered
    next_state=replace(state,pose_centered=pose.detach(),previous_pose=state.pose_centered,
        timestamp=float(timestamp),frame_id=state.frame_id+1,memory=memory.detach())
    output={k:v[0] if isinstance(v,torch.Tensor) and v.ndim else v for k,v in result.items()}
    output.update(pose_centered=pose,pose_original=original_pose(pose,torch.as_tensor(state.mesh['center'],device=pose.device)),status=status)
    return output,next_state
