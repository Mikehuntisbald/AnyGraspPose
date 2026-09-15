import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_alignment_ablation import local_rotation, summarize_pair
from lip.geometry.so3 import exp


def test_camera_frame_rotation_error_reduction():
    parent=exp(torch.tensor([[.3,.1,-.2]],dtype=torch.float64))
    correction=torch.tensor([[0.,0.,np.deg2rad(5)]],dtype=torch.float64)
    after=exp(correction)@parent
    target=exp(correction*2)@parent
    result=local_rotation(parent,after,target,correction)
    np.testing.assert_allclose(result['before_deg'],10,atol=1e-9)
    np.testing.assert_allclose(result['after_deg'],5,atol=1e-9)
    np.testing.assert_allclose(result['reduction_deg'],5,atol=1e-9)
    np.testing.assert_allclose(result['cosine'],1,atol=1e-9)


def test_wrong_direction_and_zero_step():
    parent=torch.eye(3,dtype=torch.float64)[None].repeat(2,1,1)
    target=exp(torch.tensor([[0.,.2,0.],[0.,.2,0.]],dtype=torch.float64))
    correction=torch.tensor([[0.,-.1,0.],[0.,0.,0.]],dtype=torch.float64)
    result=local_rotation(parent,exp(correction)@parent,target,correction)
    assert result['reduction_deg'][0]<0 and result['cosine'][0]<0
    assert result['reduction_deg'][1]==0 and not result['direction_valid'][1]


def test_paired_intervention_sign_and_object_macro():
    values=np.array([[100.,0.],[0.,0.],[0.,0.]])
    objects=np.array([1,2,2]); sequences=np.array(['a','b','b']); clusters=np.array(['a','b'])
    weights=np.ones((20,2),int)
    result=summarize_pair(values,objects,sequences,clusters,weights,('on','off'))
    assert result['delta']==50 and result['ci95']==[50,50]
    reverse=summarize_pair(values[:,::-1],objects,sequences,clusters,weights,('off','on'))
    assert reverse['delta']==-50 and reverse['ci95']==[-50,-50]
