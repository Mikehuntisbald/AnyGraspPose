import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from val_non_gt_common import NativeValGuard,first_legal_for_object,validate_initializers
from score_val_non_gt import summarize
from score_val_non_gt import verify_method_policy
from val_non_gt_common import NativeSplitInferenceGuard


def pose():
    p=np.eye(4);p[2,3]=.6;return p.tolist()


def test_val_guard_blocks_other_split_pixels_and_all_annotation_paths(tmp_path):
    raw=tmp_path/'raw';index=tmp_path/'index';fp=tmp_path/'fp'
    for p in (raw,index,fp):p.mkdir()
    (index/'streams.jsonl').write_text(json.dumps({'pose_cache':'poses.npz'})+'\n')
    guard=NativeValGuard(raw,index,fp,[dict(relative_dir='val/sequence/camera')])
    guard.check(raw/'val/sequence/camera/color_000000.jpg',native=True)
    for path in (raw/'test/sequence/camera/color_000000.jpg',raw/'bop/s0/test/scene_camera.json',
                 raw/'val/sequence/camera/labels_000000.npz',index/'poses.npz',fp/'weights.pt'):
        with pytest.raises(PermissionError):guard.check(path)
    assert guard.snapshot()['counts']['denied_non_val_access']==2


def test_initializer_uses_legal_class_confidence_and_missing_is_explicit():
    bad=np.eye(4).tolist();candidates=[dict(object_id=1,score=.99,pose_original=bad),
        dict(object_id=2,score=.99,pose_original=pose()),dict(object_id=1,score=.2,pose_original=pose())]
    assert first_legal_for_object(candidates,1)['score']==.2
    assert first_legal_for_object(candidates,3) is None


def test_non_gt_manifest_requires_provenance_and_keeps_missing_streams():
    streams=[dict(stream_id='a',num_frames=10),dict(stream_id='b',num_frames=10)]
    audit=dict(split_hash='s',mesh_hash='m');value=dict(completed=True,split='val',uses_gt_pose=False,fp_calls=0,**audit,
        backend='real_posecnn',checkpoint_sha256='weight',inference_source_sha256='source',
        initializers={'a':dict(frame_index=3,score=.5,pose_original=pose()),'b':None})
    assert validate_initializers(value,streams,audit) is value
    with pytest.raises(ValueError):validate_initializers(dict(value,uses_gt_pose=True),streams,audit)
    with pytest.raises(ValueError):validate_initializers(dict(value,initializers={'a':value['initializers']['a']}),streams,audit)
    with pytest.raises(ValueError):validate_initializers(dict(value,checkpoint_sha256=''),streams,audit)


def test_missing_predictions_remain_failures_in_primary_macro_denominator():
    rows=[dict(object_id=1,pose_centered=pose(),add_01=True,adds_01=True,adds_005=True),
          dict(object_id=1,pose_centered=None,add_01=False,adds_01=False,adds_005=False),
          dict(object_id=2,pose_centered=None,add_01=False,adds_01=False,adds_005=False)]
    report=summarize(rows)
    assert report['frames']==3 and report['missing_pose_frames']==2 and report['adds_005']==25.


def test_fp_baseline_permission_does_not_relax_gt_split_or_write_guards(tmp_path):
    import os
    raw=tmp_path/'raw';index=tmp_path/'index';fp=tmp_path/'fp'
    for p in (raw,index,fp):p.mkdir()
    (index/'streams.jsonl').write_text(json.dumps(dict(pose_cache='gt.npz'))+'\n')
    streams=[dict(split='val',relative_dir='val/cam')]
    guard=NativeSplitInferenceGuard(raw,index,fp,streams,'val',allow_foundationpose=True)
    guard.check(fp/'weights/model.pt');guard.check(raw/'val/cam/color_000001.jpg',native=True)
    for path,flags in [(index/'gt.npz',os.O_RDONLY),(raw/'val/cam/labels_000001.npz',os.O_RDONLY),
                       (raw/'train/cam/color_000001.jpg',os.O_RDONLY),(fp/'weights/model.pt',os.O_WRONLY)]:
        with pytest.raises(PermissionError):guard.check(path,flags)
    strict=NativeSplitInferenceGuard(raw,index,fp,streams,'val')
    with pytest.raises(PermissionError):strict.check(fp/'weights/model.pt')
    with pytest.raises(ValueError):NativeSplitInferenceGuard(raw,index,fp,streams,'train')
    assert guard.snapshot()['counts']['authorized_fp_read_opens']==1


def test_scorer_fp_opt_in_requires_exact_tracking_policy():
    m=dict(gt_pose_reads=0,gt_mask_reads=0,hand_annotation_reads=0,architecture_id='foundationpose_tracking_v1',
        frames=10,missing_pose_frames=2,initialized_streams=1,fp_calls=7,fp_iterations=2,fp_initialization_calls=0,
        access_audit=dict(foundationpose_reads_authorized=True))
    verify_method_policy(m,True)
    with pytest.raises(AssertionError):verify_method_policy(m)
    for changed in [dict(fp_calls=8),dict(fp_initialization_calls=1),dict(gt_pose_reads=1),dict(architecture_id='stream_dual_cross_residual')]:
        with pytest.raises(AssertionError):verify_method_policy(dict(m,**changed),True)
