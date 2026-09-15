import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_spatial_alignment_factorial import estimate,CONTRASTS


def test_factorial_interaction_and_removal_signs():
    names=['residual','s0_l1','s0_l0','s1_l1','s1_l0','fp']
    values=np.array([[0.,1.,3.,5.,10.,0.],[0.,1.,3.,5.,10.,0.]])
    result=estimate(values,np.array([1,2]),np.array(['a','b']),np.array(['a','b']),np.ones((100,2)),names)
    effects=result['effects']
    assert effects['spatial_with_latent']['delta']==4
    assert effects['spatial_without_latent']['delta']==7
    assert effects['remove_latent_without_spatial']['delta']==2
    assert effects['remove_latent_with_spatial']['delta']==5
    assert effects['interaction']['delta']==3 and effects['interaction']['ci99']==[3,3]
    assert len(CONTRASTS)==5


def test_no_interaction_for_additive_effects():
    names=['residual','s0_l1','s0_l0','s1_l1','s1_l0','fp']
    values=np.array([[0.,10.,12.,15.,17.,0.]])
    result=estimate(values,np.array([1]),np.array(['a']),np.array(['a']),np.ones((100,1)),names)
    assert result['effects']['interaction']['delta']==0
