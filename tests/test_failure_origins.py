import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_failure_origins import classify_entry
from diagnose_occlusion_events import extract_episodes


def rows(vis,success=None):
    return [dict(stream_id='s',frame_index=i,object_id=1,visibility=v,initialization=i==0,
        adds_005=True if success is None else success[i],status='initialized' if i==0 else 'ok') for i,v in enumerate(vis)]


def test_initial_image_does_not_count_as_observed_history():
    data=rows([.9,.9,.9,.2,.2]);event=extract_episodes(data)[0];out=classify_entry(event,data)
    assert out['category']=='no_prior_three_clear_observations' and out['initial_image_clear'] and out['starts_within_first8']


def test_inaccurate_clear_entry_is_distinct_from_missing_clear_observations():
    data=rows([.9,.9,.9,.9,.2],[True,True,False,True,False]);out=classify_entry(extract_episodes(data)[0],data)
    assert out['category']=='clear_but_inaccurate'


def test_previous_clear_run_before_an_interruption_is_retained():
    data=rows([.9,.9,.9,.9,None,.2,.2]);out=classify_entry(extract_episodes(data)[0],data)
    assert out['category']=='interrupted_clear_context'
