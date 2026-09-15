import numpy as np
from compare_startup_replications import aggregate, replicate_contrast


def test_object_macro_uses_frame_counts_inside_objects():
    sums=np.array([[[90.,0.],[0.,0.]],[[0.,10.],[0.,1.]]])
    counts=np.array([[90.,0.],[10.,1.]])
    point=aggregate(sums,counts,np.ones((1,2)))[0]
    np.testing.assert_allclose(point,[.45,.55])
    # Doubling one camera cluster changes that object's denominator, rather
    # than treating each grouped row or each object as equally many frames.
    point=aggregate(sums,counts,np.array([[2.,1.]]))[0]
    np.testing.assert_allclose(point,[90/190, (10/190+1)/2])


def test_joint_seed_draw_preserves_pair_and_common_population_draw():
    points=np.array([[1.,2.],[10.,12.],[100.,104.]])
    shared=np.array([-5.,0.,5.,20.])
    draws=points[:,None,:]+shared[None,:,None]
    weights=np.array([[3,0,0],[0,3,0],[0,0,3],[1,1,1]])
    result=replicate_contrast(points,draws,[-1,1],weights)
    assert result['per_seed']==[1.,2.,4.]
    np.testing.assert_allclose(result['paired_population_ci95'],[7/3,7/3])
    np.testing.assert_allclose(result['exploratory_seed_and_population_ci95'],np.quantile([1,2,4,7/3],[.025,.975]))
    assert result['seed_range']==[1.,4.]
