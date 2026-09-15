from pathlib import Path
import sys
import json
import os
import subprocess
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from standalone_bop_state import InferenceAccessGuard,prime_initial_observation,cache_summary,standalone_methods
from streaming_bop_utils import clear_feature_history


def test_standalone_methods_can_omit_unused_ablation_without_enabling_fp():
    assert standalone_methods({'methods':['lip_temporal']})==('lip_temporal',)
    assert standalone_methods({})==('lip_temporal','lip_no_feature_history')
    for methods in ([],['fp_tracking'],['lip_temporal','lip_temporal'],['lip_temporal','lip_fp_temporal']):
        with pytest.raises(ValueError):standalone_methods({'methods':methods})


def test_guard_blocks_cached_poses_and_native_annotation_reads(tmp_path):
    raw=tmp_path/'raw';index=tmp_path/'index';fp=tmp_path/'fp'
    for folder in (raw,index,fp):folder.mkdir()
    (index/'streams.jsonl').write_text(json.dumps({'pose_cache':'pose.npz'})+'\n')
    guard=InferenceAccessGuard(raw,index,fp)
    for path in (index/'pose.npz',raw/'labels_000000.npz',raw/'bop/s0/test/000001/scene_gt.json',
                 raw/'bop/s0/test/000001/mask/000000.png',raw/'mano/subject.pkl',fp/'weights.pt'):
        with pytest.raises(PermissionError):guard('open',(str(path),'r',os.O_RDONLY))
    with pytest.raises(PermissionError):guard.check(raw/'mask_visib/0.png',native=True)
    with pytest.raises(PermissionError):guard.check(raw/'models/object.obj',os.O_WRONLY)
    guard.check(raw/'rgb/000000.jpg',native=True)
    guard.check(index/'audit.json');guard.check(index/'meshes/model.npz')
    assert guard.snapshot()['counts']['validated_native_image_reads']==1
    assert guard.snapshot()['counts']['denied_cached_pose_access']==1


def test_python_audit_hook_blocks_actual_cached_pose_open(tmp_path):
    raw=tmp_path/'raw';index=tmp_path/'index';fp=tmp_path/'fp'
    for folder in (raw,index,fp):folder.mkdir()
    (index/'streams.jsonl').write_text(json.dumps({'pose_cache':'pose.npz'})+'\n')
    (index/'pose.npz').write_text('SYNTHETIC CANARY')
    script='''
import json,sys
from pathlib import Path
from standalone_bop_state import InferenceAccessGuard
guard=InferenceAccessGuard(*sys.argv[1:]);sys.addaudithook(guard)
try: (Path(sys.argv[2])/'pose.npz').read_bytes()
except PermissionError: print(json.dumps(guard.snapshot()))
else: raise RuntimeError('Cached pose was not blocked')
'''
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'tools'))
    result=subprocess.run([sys.executable,'-c',script,str(raw),str(index),str(fp)],env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['counts']['denied_cached_pose_access']==1


@pytest.mark.parametrize('architecture',('fixed','adaptive','rotation_anchor'))
def test_initial_rgbd_prime_preserves_external_pose_and_reference_then_clears_only_features(architecture):
    from test_stream_geometry import fixture
    from lip.models.stream_pose_reference import PoseReferenceRKTracker
    from lip.models.stream_adaptive_reference import AdaptiveReferenceRKTracker
    from lip.models.stream_rotation_anchor import RotationAnchorRKTracker
    from lip.geometry.renderer import Renderer
    from lip.geometry.so3 import center_pose,original_pose
    torch.manual_seed(7)
    model={'fixed':PoseReferenceRKTracker,'adaptive':AdaptiveReferenceRKTracker,
           'rotation_anchor':RotationAnchorRKTracker}[architecture](dense_side=2).eval()
    torch.nn.init.normal_(model.head[-1].weight,std=.02)
    model.pose_reference_feedback.readout[-1].bias.data.fill_(.3)
    if architecture!='fixed':model.reference_writer.readout[-1].bias.data.fill_(.5)
    if architecture=='rotation_anchor':model.rotation_anchor_readout.readout[-1].bias.data.fill_(.5)
    mesh,pose,k=fixture();renderer=Renderer('cpu');outputs=[]
    def observe(state):
        result,next_state=model.step(torch.zeros(3,64,64),torch.full((1,64,64),.6),10.,state,
                                     renderer=renderer,precision='fp32',image_size=32)
        assert result['status']=='ok';outputs.append(result['pose_centered'])
        return model.commit(result,next_state),result['pose_original'],'ok',0
    with torch.no_grad():
        state,status=prime_initial_observation(model,pose,mesh,k,'fixture',10.,30.,(64,64),1,1,observe)
    assert status=='ok' and state.frame_id==1 and state.timestamp==10.
    assert state.previous_pose is None and state.previous_timestamp is None
    expected=center_pose(pose,torch.as_tensor(mesh['center']))
    assert torch.equal(state.pose_centered,expected) and not torch.equal(outputs[0],expected)
    torch.testing.assert_close(original_pose(state.pose_centered,torch.as_tensor(mesh['center'])),pose,atol=1e-7,rtol=0)
    summary=cache_summary(state.cache)
    assert summary['recent_frames']==summary['context_frames']==1
    assert summary['valid_anchors']==1 and summary['dense_tokens']==4 and summary['has_pose_reference']
    reference=state.cache.reference;cleared=clear_feature_history(state)
    if architecture=='rotation_anchor':
        assert torch.equal(state.cache.initial_rotation,expected[None,:3,:3])
        assert cleared.cache.initial_rotation is state.cache.initial_rotation
    summary=cache_summary(cleared.cache)
    assert summary['recent_frames']==summary['context_frames']==summary['valid_anchors']==summary['dense_tokens']==0
    assert cleared.cache.reference is reference and cleared.pose_centered is state.pose_centered
    assert cleared.timestamp==state.timestamp and cleared.frame_id==state.frame_id
