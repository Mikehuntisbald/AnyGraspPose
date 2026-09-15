from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_reference_writes import trace_stream,summarize


def rows():
    return [dict(stream_id='s',object_id=1,frame_index=i,initialization=i==0,adds_005=True,visibility=.2,status='ok',
        reference_write_rotation_coefficient=.1,reference_write_center_coefficient=.25,reference_write_valid_target=True) for i in range(4)]


def test_reference_trace_uses_previous_writes_and_holds_failed_updates():
    source=rows();trace=trace_stream(source,.25)
    assert [r['initial_center_weight_before'] for r in trace]==[1.,.75,.5625]
    assert trace[-1]['initial_center_weight_after']==.421875
    source[2]['status']='invalid_input';del source[2]['reference_write_center_coefficient']
    trace=trace_stream(source,.25);assert trace[2]['initial_center_weight_before']==.75
    assert summarize(trace,.25)['usable']==2


def test_reference_trace_rejects_invalid_fractions_and_duplicate_frames():
    source=rows();source[1]['reference_write_center_coefficient']=.3
    with pytest.raises(ValueError,match='fraction'):trace_stream(source,.25)
    source=rows();source[1]['reference_write_valid_target']=False
    with pytest.raises(ValueError,match='Invalid target'):trace_stream(source,.25)
    source=rows();source[2]['frame_index']=1
    with pytest.raises(ValueError,match='Duplicate'):trace_stream(source,.25)
