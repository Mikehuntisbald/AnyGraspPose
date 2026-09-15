import hashlib
import json
import pytest
from lip.data.stream_sampling import load_fixed_manifest


def test_identical_manifest_can_relocate_but_reordering_is_rejected(tmp_path):
    rows=[dict(stream=0,start=1,seed=2),dict(stream=2,start=0,seed=1)]
    a=tmp_path/'a.json';a.write_text(json.dumps(rows));digest=hashlib.sha256(a.read_bytes()).hexdigest()
    b=tmp_path/'b.json';b.write_bytes(a.read_bytes())
    got,contract=load_fixed_manifest(b,digest)
    assert got==rows and contract['bound_to_config'] and contract['entries']==2
    b.write_text(json.dumps(rows[::-1]))
    with pytest.raises(ValueError,match='SHA-256'):load_fixed_manifest(b,digest)


def test_bound_manifest_cannot_silently_switch_to_online_sampling():
    with pytest.raises(ValueError,match='requires'):load_fixed_manifest(None,'a'*64)
    rows,contract=load_fixed_manifest(None)
    assert rows is None and contract['mode']=='online' and not contract['bound_to_config']


def test_unbound_legacy_manifest_does_not_claim_resume_binding(tmp_path):
    p=tmp_path/'legacy.json';p.write_text('[{"stream":0,"start":0,"seed":1}]')
    _,contract=load_fixed_manifest(p)
    assert not contract['bound_to_config'] and contract['manifest_sha256']


@pytest.mark.parametrize('payload',[[],{},[dict(stream=0,start=0,seed=True)],
    [dict(stream=0,start=-1,seed=2)],[dict(stream=0,start=0,seed=1,visibility=.1)]])
def test_invalid_or_supervision_bearing_records_rejected(tmp_path,payload):
    p=tmp_path/'bad.json';p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):load_fixed_manifest(p)
