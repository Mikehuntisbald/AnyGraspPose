"""Data access and state checks for standalone RGB-D tracking inference."""
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path


def standalone_methods(protocol):
    methods=tuple(protocol.get('methods',('lip_temporal','lip_no_feature_history')))
    if not methods or len(set(methods))!=len(methods) or not set(methods)<={'lip_temporal','lip_no_feature_history'}:
        raise ValueError('Standalone evaluation allows only distinct zero-FP LIP methods')
    return methods


class InferenceAccessGuard:
    """Python-open audit plus explicit validation around native image readers.

    This is not an OS sandbox and does not intercept arbitrary C++ syscalls.
    """
    def __init__(self, raw_root, index_root, fp_root):
        self.raw_root=Path(raw_root).resolve();self.index_root=Path(index_root).resolve()
        self.fp_root=Path(fp_root).resolve();self.counts=Counter()
        self.pose_caches={(self.index_root/row['pose_cache']).resolve()
                          for row in map(json.loads,(self.index_root/'streams.jsonl').read_text().splitlines())}

    def check(self, path, flags=os.O_RDONLY, native=False):
        if not isinstance(path,(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(path)).resolve()
        writing=bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        category=None
        if path.is_relative_to(self.fp_root):category='fp_access'
        elif path in self.pose_caches:category='cached_pose_access'
        elif path.is_relative_to(self.raw_root):
            if writing:category='raw_write'
            elif path.name.startswith(('labels_','scene_gt')):category='raw_annotation_access'
            elif any(part in ('mask','mask_visib') or part.startswith('mano') for part in path.relative_to(self.raw_root).parts):category='mask_or_hand_access'
        if category is not None:
            self.counts['denied_'+category]+=1
            raise PermissionError('Standalone inference denied '+category+': '+str(path))
        if native:
            if not path.is_relative_to(self.raw_root):raise PermissionError('Native image read must stay inside the data root')
            self.counts['validated_native_image_reads']+=1
        elif path.is_relative_to(self.raw_root):self.counts['python_raw_read_opens']+=1
        elif path.is_relative_to(self.index_root):self.counts['python_index_metadata_opens']+=1

    def __call__(self,event,args):
        if event=='open':self.check(args[0],args[2])

    def snapshot(self):
        keys=('denied_fp_access','denied_cached_pose_access','denied_raw_write',
              'denied_raw_annotation_access','denied_mask_or_hand_access',
              'validated_native_image_reads','python_raw_read_opens','python_index_metadata_opens')
        return dict(counts={key:self.counts[key] for key in keys},
                    scope='Python open events plus explicit native-image-read checks in this inference entry. Not an OS sandbox or a global interceptor of native-library file access.')


def cache_summary(cache):
    bank=getattr(cache,'anchors',None);patches=getattr(cache,'patches',None)
    valid=int(bank.valid.sum().item()) if bank is not None else 0
    return dict(recent_frames=len(cache.metadata),context_frames=len(getattr(cache,'contexts',())),
                valid_anchors=valid,dense_tokens=valid*int(patches.shape[2]) if patches is not None else 0,
                has_pose_reference=getattr(cache,'reference',None) is not None,bytes=cache.kv_bytes)


def prime_initial_observation(model,pose,mesh,K,stream_id,timestamp,fps,image_shape,obj_id,camera_id,observe):
    """Encode the first observed RGB-D, then retain the supplied external pose."""
    state=model.initialize(pose,mesh,K,stream_id,timestamp-1/fps,image_shape=image_shape,
                           object_id=obj_id,camera_id=camera_id)
    state,_,warm_status,_=observe(state)
    state=model.correct(state,pose,timestamp=timestamp)
    state=replace(state,previous_pose=None,previous_timestamp=None)
    reference=getattr(state.cache,'reference',None)
    if reference is not None:
        import torch
        assert torch.equal(reference.pose,state.pose_centered[None])
        assert float(reference.started_timestamp[0])==timestamp
        observed=getattr(reference,'observed_timestamp',None)
        assert observed is None or float(observed[0])==timestamp
    return state,warm_status
