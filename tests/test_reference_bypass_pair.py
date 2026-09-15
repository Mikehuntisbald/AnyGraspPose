import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_reference_bypass_pair import check_repeat,upstream_status


@pytest.fixture
def pair(tmp_path):
    m=dict(completed=True,population_verified=True,frames=23200,streams=list(range(320)),split='val',fp_calls=0,critic_calls=0,
        checkpoint_sha256='ckpt',source_sha256='src',split_hash='split',mesh_hash='mesh',initial_poses_sha256='init',config={'precision':'bf16'})
    rows=[dict(stream_id=str(i%320),frame_index=i//320,initialization=i<320,object_id=1,visibility=.2,status='ok',
        add_01=True,adds_005=True,pose_centered=[[1.,0.],[0.,1.]]) for i in range(23200)]
    folders=[tmp_path/'archived',tmp_path/'repeat']
    for folder in folders:
        folder.mkdir();(folder/'manifest.json').write_text(json.dumps(m))
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return folders,rows,m


def test_frozen_repeat_accepts_identical_poses_and_rejects_changed_trajectory(pair):
    folders,rows,m=pair
    result=check_repeat(*folders)
    assert result['changed_pose_rows']==0 and result['max_pose_element_abs_difference']==0
    rows[1000]['pose_centered'][0][1]=.01
    (folders[1]/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='tolerance'):check_repeat(*folders)


def test_frozen_repeat_rejects_protocol_changes_and_duplicate_frames(pair):
    folders,rows,m=pair
    changed=copy.deepcopy(m);changed['config']['precision']='fp32'
    (folders[1]/'manifest.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='provenance'):check_repeat(*folders)
    (folders[1]/'manifest.json').write_text(json.dumps(m));rows[-1]=rows[0]
    (folders[1]/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Duplicate'):check_repeat(*folders)


def test_completed_upstream_does_not_require_a_live_former_controller(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'phase':'completed'}))
    (tmp_path/'comparison.json').write_text(json.dumps({'completed':True}))
    assert upstream_status(tmp_path,999999999)==('completed',None)
    assert upstream_status(tmp_path,999999999,controller_script='run_spatial_memory.py')==('completed',None)
    with pytest.raises(ValueError,match='Unsupported'):upstream_status(tmp_path,999999999,controller_script='unknown.py')
    (tmp_path/'comparison.json').write_text(json.dumps({'completed':False}))
    with pytest.raises(RuntimeError,match='comparison incomplete'):upstream_status(tmp_path,999999999)


def test_running_upstream_requires_exact_live_process_and_failure_is_terminal(tmp_path):
    import os
    (tmp_path/'status.json').write_text(json.dumps({'phase':'training'}))
    with pytest.raises(RuntimeError,match='identity'):upstream_status(tmp_path,os.getpid())
    (tmp_path/'status.json').write_text(json.dumps({'phase':'failed'}))
    with pytest.raises(RuntimeError,match='factorial failed'):upstream_status(tmp_path,999999999)
