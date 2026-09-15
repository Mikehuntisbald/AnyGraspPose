import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('events',Path(__file__).resolve().parents[1]/'tools/diagnose_occlusion_events.py')
events=importlib.util.module_from_spec(spec);spec.loader.exec_module(events)


def row(i,v=.8,stream='a',initial=False,good=True,status='ok'):
    return dict(stream_id=stream,frame_index=i,visibility=v,initialization=initial,object_id=1,adds_005=good,adds_01=good,status=status)


def test_initialization_does_not_count_as_three_clear_tracking_frames():
    rs=[row(0,initial=True),row(1),row(2)]+[row(i,.2) for i in range(3,12)]+[row(i) for i in range(12,22)]
    e=events.extract_episodes(rs)[0]
    assert e['pre_frames']==[1,2] and not e['pre_complete']
    assert e['length']==9 and e['post_complete'] and e['censor_reason'] is None


def test_missing_visibility_and_frame_gaps_split_events_and_censor_post_window():
    rs=[row(i) for i in range(3)]+[row(3,.2),row(4,.2),row(5,None),row(6,.2),row(8,.2),row(9)]
    es=events.extract_episodes(rs)
    assert [e['frames'] for e in es]==[[3,4],[6],[8]]
    assert [e['censor_reason'] for e in es]==['unknown_visibility','frame_gap','stream_end']
    assert es[0]['pre_complete'] and not es[1]['pre_complete'] and not es[2]['pre_complete']


def test_recovery_requires_three_consecutive_actual_valid_predictions():
    rs=[row(10),row(11,status='invalid'),row(12),row(13),row(14)]
    assert events.stable_recovery(rs)==5
    assert events.stable_recovery([row(1),row(2),row(4)]) is None
    assert events.stable_recovery([row(0,initial=True),row(1),row(2)]) is None


def test_streams_are_isolated_and_duplicate_rows_fail():
    es=events.extract_episodes([row(0,.2),row(1,.2)]+[row(i,stream='b') for i in range(20)])
    assert len(es)==1 and es[0]['post_frames']==[] and es[0]['censor_reason']=='stream_end'
    try:events.extract_episodes([row(1),row(1)])
    except ValueError:pass
    else:raise AssertionError('Duplicate frame accepted')
