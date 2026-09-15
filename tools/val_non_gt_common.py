"""Native s0-val registry, provenance and inference-only data boundaries."""
import json
import os
from pathlib import Path
import numpy as np
from standalone_bop_state import InferenceAccessGuard
from streaming_bop_utils import legal_pose


def val_streams(index_root):
    return sorted([s for s in map(json.loads,(Path(index_root)/'streams.jsonl').read_text().splitlines())
                   if s['split']=='val'],key=lambda s:s['stream_id'])


class NativeValGuard(InferenceAccessGuard):
    def __init__(self,raw_root,index_root,fp_root,streams):
        super().__init__(raw_root,index_root,fp_root)
        self.allowed={(self.raw_root/s['relative_dir']).resolve() for s in streams}

    def check(self,path,flags=os.O_RDONLY,native=False):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(path)).resolve()
            image=p.name.startswith(('color_','aligned_depth_to_color_'))
            if p.is_relative_to(self.raw_root) and (p.is_relative_to(self.raw_root/'bop') or (image and p.parent not in self.allowed)):
                self.counts['denied_non_val_access']+=1;raise PermissionError('Non-GT val cannot read other splits: '+str(p))
        super().check(path,flags,native)

    def snapshot(self):
        result=super().snapshot();result['counts']['denied_non_val_access']=self.counts['denied_non_val_access'];return result


class NativeSplitInferenceGuard(InferenceAccessGuard):
    """Explicit train/val pixel boundary; FP reads require a baseline opt-in."""
    def __init__(self,raw_root,index_root,fp_root,streams,split,allow_foundationpose=False):
        if split not in ('train','val') or not streams or any(s['split']!=split for s in streams):
            raise ValueError('Native inference must bind one explicit train/val split')
        super().__init__(raw_root,index_root,fp_root)
        if self.fp_root.is_relative_to(self.raw_root) or self.fp_root.is_relative_to(self.index_root):
            raise ValueError('FP code cannot contain dataset or index resources')
        self.allowed={(self.raw_root/s['relative_dir']).resolve() for s in streams}
        self.split=split;self.allow_foundationpose=bool(allow_foundationpose)

    def check(self,path,flags=os.O_RDONLY,native=False):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(path)).resolve()
            image=p.name.startswith(('color_','aligned_depth_to_color_'))
            if p.is_relative_to(self.raw_root) and (p.is_relative_to(self.raw_root/'bop') or (image and p.parent not in self.allowed)):
                self.counts['denied_other_split_access']+=1;raise PermissionError('Image outside bound native split: '+str(p))
            if self.allow_foundationpose and p.is_relative_to(self.fp_root):
                if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
                    self.counts['denied_fp_write']+=1;raise PermissionError('FP baseline code/weights are read-only')
                if native:raise PermissionError('Native image reader cannot read FP resources')
                self.counts['authorized_fp_read_opens']+=1;return
        super().check(path,flags,native)

    def snapshot(self):
        result=super().snapshot();result.update(split=self.split,foundationpose_reads_authorized=self.allow_foundationpose)
        for key in ('denied_other_split_access','denied_fp_write','authorized_fp_read_opens'):result['counts'][key]=self.counts[key]
        return result


def first_legal_for_object(candidates,object_id):
    eligible=[row for row in candidates if row['object_id']==object_id and np.isfinite(row['score']) and legal_pose(row['pose_original'])]
    return max(eligible,key=lambda r:r['score']) if eligible else None


def validate_initializers(initials,streams,audit):
    if not initials.get('completed') or initials.get('uses_gt_pose') is not False or initials.get('fp_calls')!=0:
        raise ValueError('A completed real non-GT, zero-FP initializer is required')
    if initials.get('split')!='val' or any(initials.get(k)!=audit[k] for k in ('split_hash','mesh_hash')):
        raise ValueError('Initializer split/mesh provenance mismatch')
    if not initials.get('backend') or not initials.get('checkpoint_sha256') or not initials.get('inference_source_sha256'):
        raise ValueError('Real initializer model and inference provenance required')
    if set(initials['initializers'])!={s['stream_id'] for s in streams}:raise ValueError('Initializer stream population mismatch')
    for stream in streams:
        value=initials['initializers'][stream['stream_id']]
        if value is None:continue
        if type(value['frame_index']) is not int or not 0<=value['frame_index']<stream['num_frames']:
            raise ValueError('Initializer frame outside its stream')
        if not legal_pose(value['pose_original']) or not np.isfinite(value['score']):raise ValueError('Illegal external initializer')
    return initials
