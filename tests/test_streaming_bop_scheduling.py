import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from accelerate_streaming_bop import assign_balanced,frame_count


def test_reassignment_covers_every_remaining_stream_once_and_balances_frames():
    plans = [dict(scene_id=i,obj_id=1,targets=[dict(im_id=100+i)],initializer=dict(im_id=0)) for i in range(70)]
    plans.append(dict(scene_id=99,obj_id=1,targets=[dict(im_id=20)],initializer=None))
    assignment,loads = assign_balanced(plans,32)
    pairs = [tuple(v) for values in assignment.values() for v in values]
    assert len(pairs)==len(set(pairs))==len(plans)
    assert sum(loads)==sum(map(frame_count,plans))
    assert max(loads)-min(loads)<=max(map(frame_count,plans))
