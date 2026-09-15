from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import pytest
import torch
import json
import hashlib
from evaluate_reference_bypass import install_mode,checked_shards
from lip.models.stream_spatial_memory import SpatialRKTracker
from lip.models.stream_pose_reference import PoseReferenceRKTracker
from test_stream_rk import features,meta


def test_bypass_preserves_weights_and_matches_adapted_parent_pose():
    torch.manual_seed(31);parent=SpatialRKTracker(dense_side=2).eval()
    torch.nn.init.normal_(parent.head[-1].weight,std=.02)
    torch.nn.init.normal_(parent.patch_attention.out_proj.weight,std=.02)
    model=PoseReferenceRKTracker(dense_side=2).eval();model.load_state_dict(parent.state_dict(),strict=False)
    model.pose_reference_feedback.readout[-1].bias.data.fill_(.8)
    saved={name:t.clone() for name,t in model.state_dict().items()};f=features();a=b=None
    with torch.no_grad():
        normal,_=model(f,meta(0));expected,_=parent(f,meta(0))
        assert not torch.equal(normal['pose_centered'],expected['pose_centered'])
        handle=install_mode(model,'zero')
        for i in range(14):
            expected,a=parent(f,meta(i),a);actual,b=model(f,meta(i),b)
            assert torch.equal(expected['pose_centered'],actual['pose_centered'])
            assert actual['reference_rotation_coefficient'].count_nonzero()==0
            assert actual['reference_center_coefficient'].count_nonzero()==0
            f=dict(f,T_base_centered=actual['pose_centered'])
        assert b.reference is not None
        handle.remove()
    assert all(torch.equal(t,model.state_dict()[name]) for name,t in saved.items())
    assert install_mode(model,'learned') is None


def test_intervention_rejects_wrong_architecture_or_mode():
    with pytest.raises(ValueError,match='pose-reference'):install_mode(SpatialRKTracker(dense_side=2),'zero')
    with pytest.raises(ValueError,match='Explicit'):install_mode(PoseReferenceRKTracker(dense_side=2),'guess')


def test_merge_rejects_mixed_modes_and_changed_predictions(tmp_path):
    for rank in range(2):
        p=tmp_path/f'rank{rank}';p.mkdir();raw=b'{"test":1}\n';(p/'predictions.jsonl').write_bytes(raw)
        r=dict(completed=True,mode='zero',weights_bitwise_unchanged=True,source_and_intervention_bound=True,
            predictions_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256='weights',base_source_sha256='source',wrapper_sha256='wrapper')
        (p/'intervention.json').write_text(json.dumps(r));(p/'manifest.json').write_text(json.dumps(dict(inference_intervention=r)))
    assert len(checked_shards(tmp_path,2,'zero'))==2
    with pytest.raises(ValueError,match='mode'):checked_shards(tmp_path,2,'learned')
    folder=tmp_path/'rank1';original=json.loads((folder/'intervention.json').read_text());mixed=dict(original,mode='learned')
    (folder/'intervention.json').write_text(json.dumps(mixed));(folder/'manifest.json').write_text(json.dumps(dict(inference_intervention=mixed)))
    with pytest.raises(ValueError,match='mode'):checked_shards(tmp_path,2,'zero')
    (folder/'intervention.json').write_text(json.dumps(original));(folder/'manifest.json').write_text(json.dumps(dict(inference_intervention=original)))
    (tmp_path/'rank1/predictions.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='predictions'):checked_shards(tmp_path,2,'zero')


def test_zero_writer_keeps_feedback_and_matches_same_weight_immutable_reference():
    from test_stream_adaptive_reference import pair
    parent,model=pair();model.reference_writer.readout[-1].bias.data.fill_(.5)
    saved={n:t.clone() for n,t in model.state_dict().items()};f=features();a=b=None
    handle=install_mode(model,'zero','writer')
    with torch.no_grad():
        for i in range(18):
            expected,a=parent(f,meta(i),a);actual,b=model(f,meta(i),b)
            assert torch.equal(expected['pose_centered'],actual['pose_centered'])
            assert actual['reference_write_rotation_coefficient'].count_nonzero()==0
            assert actual['reference_write_center_coefficient'].count_nonzero()==0
            assert actual['reference_rotation_coefficient'].abs().sum()>0
            f=dict(f,T_base_centered=actual['pose_centered'])
    handle.remove();assert all(torch.equal(t,model.state_dict()[n]) for n,t in saved.items())
    with pytest.raises(ValueError,match='adaptive'):install_mode(parent,'zero','writer')


def test_merge_distinguishes_feedback_and_writer_components(tmp_path):
    p=tmp_path/'rank0';p.mkdir();raw=b'{}\n';(p/'predictions.jsonl').write_bytes(raw)
    r=dict(completed=True,mode='zero',component='writer',weights_bitwise_unchanged=True,source_and_intervention_bound=True,
        predictions_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256='ckpt',base_source_sha256='src',wrapper_sha256='wrapper')
    (p/'intervention.json').write_text(json.dumps(r));(p/'manifest.json').write_text(json.dumps(dict(inference_intervention=r)))
    assert checked_shards(tmp_path,1,'zero','writer')
    with pytest.raises(ValueError,match='component'):checked_shards(tmp_path,1,'zero','feedback')


@pytest.mark.parametrize('channel',('rotation','center'))
def test_partial_writer_preserves_selected_reference_component_and_other_write(channel):
    from test_stream_adaptive_reference import pair
    _,model=pair();model.reference_writer.readout[-1].bias.data.fill_(.5)
    saved={name:t.clone() for name,t in model.state_dict().items()}
    f=features();initial=f['T_base_centered'].clone();cache=None;active=[]
    handle=install_mode(model,'zero','writer',channel)
    with torch.no_grad():
        for i in range(18):
            result,cache=model(f,meta(i),cache)
            if channel=='rotation':
                assert torch.equal(cache.reference.pose[:,:3,:3],initial[:,:3,:3])
                assert not result['reference_write_rotation_coefficient'].count_nonzero()
                active.append(float(result['reference_write_center_norm'].sum()))
            else:
                assert torch.equal(cache.reference.pose[:,:3,3],initial[:,:3,3])
                assert not result['reference_write_center_coefficient'].count_nonzero()
                active.append(float(result['reference_write_rotation_norm'].sum()))
            assert result['reference_rotation_coefficient'].abs().sum()>0
            f=dict(f,T_base_centered=result['pose_centered'])
    handle.remove()
    assert max(active)>0
    assert all(torch.equal(t,model.state_dict()[name]) for name,t in saved.items())
    with pytest.raises(ValueError,match='Partial'):install_mode(model,'learned','writer',channel)
    with pytest.raises(ValueError,match='Partial'):install_mode(model,'zero','feedback',channel)


def test_merge_rejects_mixed_partial_writer_channels(tmp_path):
    p=tmp_path/'rank0';p.mkdir();raw=b'{}\n';(p/'predictions.jsonl').write_bytes(raw)
    r=dict(completed=True,mode='zero',component='writer',zero_channels='rotation',weights_bitwise_unchanged=True,
        source_and_intervention_bound=True,predictions_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoint_sha256='ckpt',base_source_sha256='src',wrapper_sha256='wrapper')
    (p/'intervention.json').write_text(json.dumps(r));(p/'manifest.json').write_text(json.dumps(dict(inference_intervention=r)))
    assert checked_shards(tmp_path,1,'zero','writer','rotation')
    with pytest.raises(ValueError,match='channels'):checked_shards(tmp_path,1,'zero','writer','center')
    with pytest.raises(ValueError,match='channels'):checked_shards(tmp_path,1,'zero','writer')
