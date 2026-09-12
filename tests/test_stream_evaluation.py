import json
import numpy as np
from lip.evaluate_stream import save_reports
from lip.evaluation.metrics import errors
from test_stream_geometry import fixture


def test_stream_fp32_metrics_and_intersection_subsets(tmp_path):
    mesh,t,k=fixture();p=t.clone();p[0,3]+=.01
    a=errors(p,t,mesh['vertices'],.15);b=errors(p,t,mesh['vertices'],.15,dtype='f4')
    assert abs(a['center_mm']-b['center_mm'])<1e-5 and abs(a['add_m']-b['add_m'])<1e-6
    rows=[]
    for i,(moving,visibility) in enumerate([(True,.2),(True,.2),(False,.2),(True,.5)]):
        rows.append(dict(**b,stream_id='subject/sequence/camera',object_id=1,camera_id='c',frame_index=i,
            initialization=i==0,moving=moving,visibility=visibility,visibility_bin='[.1,.3)' if visibility<.3 else '[.3,.6)',
            lost=False,status='initialized' if i==0 else 'ok'))
    save_reports(tmp_path,rows);report=json.loads((tmp_path/'metrics.json').read_text())
    assert report['micro']['count']==4 and report['excluding_initialization']['micro']['count']==3
    assert report['moving_and_visibility_lt_03']['micro']['count']==1
    assert report['per_camera']['c']['micro']['count']==3
