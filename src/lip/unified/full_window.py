"""Full-video TRAINING windows; native bytes and explicit empirical-error transport.

GT is used to construct a training pose perturbation, never a deployment reset.
No rendered, encoded, or pose-conditioned feature is cached here.
"""
import cv2
import numpy as np
import torch
from lip.engine.jepa_cuda_replay import CUDA_CAPTURE_LOCK


def choose_start(seed,frames,count):
    if not 0<count<=frames:raise ValueError('Invalid full-window length')
    # Keep stream, donor and perturbation RNG draws identical to the old sampler.
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),33001]))
    return int(rng.integers(frames-count+1))


@torch.no_grad()
def transport_centered_error(estimate,source_truth,target_truth):
    """Preserve camera-frame rotation error and object-center translation error."""
    result=target_truth.clone()
    rotation_error=estimate[..., :3,:3]@source_truth[..., :3,:3].transpose(-1,-2)
    result[..., :3,:3]=rotation_error@target_truth[..., :3,:3]
    result[..., :3,3]=target_truth[..., :3,3]+estimate[..., :3,3]-source_truth[..., :3,3]
    return result


def decode_frame(factory,sid,stream,frame,entries):
    names=(f'color_{frame:06d}.jpg',f'aligned_depth_to_color_{frame:06d}.png',f'labels_{frame:06d}.npz')
    if entries is not None and all(name in entries for name in names):
        return factory.packed.decode(entries,frame,stream['object_id']),True
    directory=factory.root/stream['relative_dir']
    rgb=cv2.imread(str(directory/names[0]));depth=cv2.imread(str(directory/names[1]),cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:raise FileNotFoundError(f'Missing full-window native frame: {sid}/{frame}')
    with np.load(directory/names[2],allow_pickle=False) as labels:mask=(labels['seg']==stream['object_id']).copy()
    return (cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB).transpose(2,0,1),depth[None],mask[None]),False


def prepare_cpu(factory,seed,count):
    if factory.split!='train':raise ValueError('Full-window error transport is training-only')
    rng=np.random.default_rng(seed)
    sid,initial=factory.options[int(rng.integers(len(factory.options)))];stream=factory.streams[sid]
    first=choose_start(seed,stream['num_frames'],count)
    entries=None
    if factory.packed is not None:
        cached=set(factory.packed.records[sid]['frames'])
        if any(frame in cached for frame in range(first,first+count)):entries=factory.packed.stream(sid)
    decoded=list(factory.pool.map(lambda frame:decode_frame(factory,sid,stream,frame,entries),range(first,first+count)))
    if sid not in factory.pose_labels:
        with np.load(factory.index/stream['pose_cache'],allow_pickle=False) as z:factory.pose_labels[sid]=z['poses'].copy()
    arrays=(np.stack([x[0][0] for x in decoded]),np.stack([x[0][1] for x in decoded]).astype('f4'),
        np.stack([x[0][2] for x in decoded]),factory.pose_labels[sid][first:first+count])
    if len(arrays[-1])!=count:raise ValueError('Full-window labels cross the stream boundary')
    with CUDA_CAPTURE_LOCK,torch.cuda.device(factory.cuda_device):
        arrays=tuple(torch.from_numpy(np.ascontiguousarray(a)).pin_memory() for a in arrays)
    provenance=dict(kind='uniform_full_training_window',native_initializer_frame=int(initial['frame_index']),
        first_frame=first,count=count,stream_frames=stream['num_frames'],cached_frames=sum(x[1] for x in decoded),
        raw_native_frames=sum(not x[1] for x in decoded),pose_source='Training-only transport of empirical native initialization error; not native current-frame initialization')
    return (sid,initial,first,count,rng,arrays),provenance
