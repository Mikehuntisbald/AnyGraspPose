import numpy as np
from lip.data.basin_targets import fields_from_metrics,candidate_outcomes

def row(e):return dict(add_m=e,adds_m=e,rotation_deg=e*10,center_mm=e*1000)

def test_successes_keep_different_quality_and_signed_gain():
 b=[row(.040),row(.090),row(.040),row(.3)];a=[row(.012),row(.049),row(.050),row(.2)]
 r=fields_from_metrics(b,a,1.)
 assert r['y_converge'].tolist()==[1,1,1,0]
 assert r['y_improve'].tolist()==[1,1,0,1]
 assert np.allclose(r['delta_e'],[.028,.041,-.010,.100])
 assert r['e_after'][0]!=r['e_after'][1]

def test_strict_threshold_and_margin():
 before=.04;margin=.01
 r=fields_from_metrics([row(before),row(.2)],[row(before-margin),row(.1)],1.,margin=margin)
 assert r['y_improve'][0]==0 and r['y_converge'][1]==0

def test_metric_units_and_pose_coordinates():
 v=np.array([[-.05,0,0],[.05,0,0],[0,.04,0]],dtype='f4');gt=np.eye(4,dtype='f4');gt[2,3]=.7
 pred=gt.copy();pred[0,3]+=.02;after=gt.copy();after[0,3]+=.01
 center=np.array([.03,.02,0],dtype='f4');r=candidate_outcomes(pred[None],after[None],gt,v,center,.1)
 assert np.allclose(r['add_before_over_d'],.2,atol=1e-6)
 assert np.allclose(r['center_after_mm'],10.,atol=1e-5)
 assert np.allclose(r['candidate_pose_original'][0,:3,3],pred[:3,3]-center)
 assert r['candidate_pose'].dtype==np.float32
