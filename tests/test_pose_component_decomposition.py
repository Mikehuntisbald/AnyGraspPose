import numpy as np
from decompose_pose_components import component_errors, contributions


def test_rotation_center_replacement_uses_centered_camera_translation():
    points=np.array([[-.1,0,0],[.1,0,0],[0,.03,0]],dtype='f4')
    gt=np.eye(4,dtype='f4');gt[:3,3]=[.2,-.1,.6]
    on=gt.copy();off=gt.copy();off[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];off[0,3]+=.02
    values=component_errors(points,gt,on,off)
    assert values[0]==0
    np.testing.assert_allclose(values[2],.02,atol=1e-7)
    swapped=component_errors(points,gt,off,on)
    np.testing.assert_allclose(swapped,values[[3,2,1,0]],atol=1e-7)
    effect=contributions(values)
    np.testing.assert_allclose(effect['rotation_contribution']+effect['center_contribution'],values[3]-values[0])


def test_threshold_interaction_is_shared_between_replacement_orders():
    # Neither component alone crosses the success threshold; both do.
    effects=contributions(np.array([0.,0.,0.,100.]))
    assert effects=={'off_minus_on':100.,'rotation_contribution':50.,'center_contribution':50.}
