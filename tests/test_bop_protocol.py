import csv
import numpy as np
from lip.benchmark.bop_io import csv_row,load_predictions,select_initializers,FIELDS

def test_original_mesh_csv_units_and_rotation_round_trip(tmp_path):
    r=dict(scene_id=12,im_id=8,obj_id=3,score=.7);t=np.eye(4);t[:3,3]=[.12,-.03,.75]
    row=csv_row(r,t);assert np.allclose(np.fromstring(row['t'],sep=' '),[120,-30,750])
    p=tmp_path/'pose.csv'
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerow(row)
    actual=load_predictions(p)[0];np.testing.assert_allclose(actual['pose'],t,rtol=0,atol=1e-15)
    assert actual['score']==.7

def test_target_selection_keeps_missing_denominator_and_uses_score_only():
    def row(scene,frame,obj,score):return dict(scene_id=scene,im_id=frame,obj_id=obj,score=score,pose=np.eye(4))
    targets=[dict(scene_id=0,im_id=0,obj_id=1,inst_count=1),dict(scene_id=0,im_id=4,obj_id=2,inst_count=1)]
    good=row(0,0,1,.8);rows=[row(0,0,1,.2),good,row(0,1,1,1),row(9,0,1,1)]
    selected,missing=select_initializers(rows,targets)
    assert len(selected)==1 and selected[0] is good
    assert missing==[dict(target=targets[1],missing=1)]
