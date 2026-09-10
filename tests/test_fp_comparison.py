import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('compare_fp',Path(__file__).parents[1]/'tools/compare_foundationpose.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(tmp_path):
    a,b=tmp_path/'lip',tmp_path/'fp';a.mkdir();b.mkdir()
    manifest=dict(split='val',mode='closed-loop',initial_pose_source='gt_first_frame',split_hash='s',mesh_hash='m',streams=['s1'],full_sequences=True,quick_subset=False,completed=True,config={'precision':'bf16'},precision='fp16')
    row=dict(stream_id='s1',frame_index=0,object_id=1,visibility=.5,visibility_bin='mid',moving=False)
    for p in [a,b]:
        (p/'manifest.json').write_text(json.dumps(manifest));(p/'predictions.jsonl').write_text(json.dumps(row)+'\n')
        (p/'metrics.json').write_text(json.dumps(dict(macro_object={'adds_01':.5},latency_seconds={})))
    return a,b

def test_comparison_population(tmp_path):
    a,b=fixture(tmp_path);assert m.compare(a,b)['matched']
    (b/'predictions.jsonl').write_text('')
    with pytest.raises(ValueError,match='population'):m.compare(a,b)

def test_comparison_rejects_protocol_and_incomplete(tmp_path):
    a,b=fixture(tmp_path);p=b/'manifest.json';r=json.loads(p.read_text());r['completed']=False;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='incomplete'):m.compare(a,b)
    r['completed']=True;r['initial_pose_source']='other';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Protocol'):m.compare(a,b)

def test_comparison_rejects_duplicate(tmp_path):
    a,b=fixture(tmp_path);p=b/'predictions.jsonl';p.write_text(p.read_text()*2)
    with pytest.raises(ValueError,match='Duplicate'):m.compare(a,b)
