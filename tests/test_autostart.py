import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location('lip_auto_full_train',Path(__file__).resolve().parents[1]/'tools/auto_full_train.py')
auto=importlib.util.module_from_spec(spec);spec.loader.exec_module(auto)


def upload_fixture(tmp_path):
    archive=tmp_path/'subject.tar.gz';archive.write_bytes(b'fixture')
    digest=hashlib.sha256(b'fixture').hexdigest()
    expected=dict(files=[dict(name=archive.name,bytes=7)],total_bytes=7)
    receipt=dict(status='complete',total_bytes=7,files=[dict(name=archive.name,bytes=7,remote_path=str(archive),
                 status='verified',sha256=digest,remote_sha256=digest,gzip_crc_verified=True)])
    return expected,receipt


def test_auto_upload_requires_complete_receipt_and_every_file(tmp_path):
    expected,receipt=upload_fixture(tmp_path)
    assert len(auto.validate_upload(expected,receipt,tmp_path)['files'])==1
    partial=copy.deepcopy(receipt);partial['status']='running'
    with pytest.raises(ValueError,match='completion'):auto.validate_upload(expected,partial,tmp_path)
    missing=copy.deepcopy(receipt);missing['files']=[]
    with pytest.raises(ValueError,match='missing'):auto.validate_upload(expected,missing,tmp_path)
    duplicate=copy.deepcopy(receipt);duplicate['files']*=2
    with pytest.raises(ValueError,match='duplicating'):auto.validate_upload(expected,duplicate,tmp_path)


def test_auto_upload_rejects_hash_path_or_size_mismatch(tmp_path):
    expected,receipt=upload_fixture(tmp_path)
    for field,value in [('remote_sha256','0'*64),('remote_path',str(tmp_path/'outside.gz')),('bytes',8),('gzip_crc_verified',False)]:
        changed=copy.deepcopy(receipt);changed['files'][0][field]=value
        with pytest.raises(ValueError):auto.validate_upload(expected,changed,tmp_path)


def test_auto_failure_stops_before_training_and_records_error(tmp_path,monkeypatch):
    flow=auto.Workflow(tmp_path,tmp_path/'runs/job',tmp_path/'cache/raw',tmp_path/'archives')
    flow.context.update(source_sha256='source',upload_receipt_sha256='data')
    # The subprocess really fails; shorten only the status polling interval.
    import time
    sleep=time.sleep
    monkeypatch.setattr(auto.time,'sleep',lambda _:sleep(.01))
    reached=[]
    with pytest.raises(RuntimeError,match='exit code 3'):
        flow.run_stage('failed_preflight',[sys.executable,'-c','raise SystemExit(3)'])
        reached.append('training')
    assert not reached
    receipt=json.loads((flow.job/'failed_preflight.json').read_text())
    assert receipt['status']=='failed' and receipt['returncode']==3
    flow.lock.close()


def test_auto_pipeline_order_never_uses_subset_training(tmp_path):
    flow=auto.Workflow(tmp_path,tmp_path/'runs/job',tmp_path/'cache/raw',tmp_path/'archives')
    flow.wait_upload=lambda:None;flow.snapshot=lambda:None;flow.wait_gpus=lambda:None
    calls=[]
    flow.run_stage=lambda name,cmd,**kw:calls.append((name,[str(x) for x in cmd],kw))
    flow.execute()
    names=[x[0] for x in calls]
    assert names.index('extract')<names.index('index')<names.index('approve_full_preflight')<names.index('train_40000')
    assert calls[names.index('train_40000')][1]==['bash','scripts/train_8gpu.sh']
    assert all('--allow-verified-subset' not in cmd for _,cmd,_ in calls)
    flow.lock.close()
