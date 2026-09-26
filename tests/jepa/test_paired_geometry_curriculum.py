import torch
from dataclasses import dataclass
from lip.unified.paired_geometry_curriculum import mirror_estimate,paired_scenes
from lip.geometry.so3 import update


def test_opposite_errors_in_the_same_camera_frame():
    truth=torch.eye(4)[None].repeat(2,1,1);truth[:,:3,3]=torch.tensor([.1,.2,.8])
    noise=torch.tensor([[.1,-.2,.15, .01,.02,-.03],[0.,0.,0.,0.,0.,0.]])
    estimate=update(truth,noise[:,:3],noise[:,3:],torch.ones(2))
    paired=mirror_estimate(estimate,truth)
    expected=update(truth,-noise[:,:3],-noise[:,3:],torch.ones(2))
    torch.testing.assert_close(paired,expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(mirror_estimate(paired,truth),estimate,atol=1e-6,rtol=1e-6)


def test_pair_preserves_real_crop_and_calibration():
    @dataclass
    class Scene:
        pose: torch.Tensor
        state: torch.Tensor
        diameter: float
        cad: dict
        k_crop: torch.Tensor
        render: dict
        rgb: torch.Tensor
        depth: torch.Tensor
        affine: torch.Tensor
    truth=torch.eye(4)[None];truth[:,2,3]=1.
    estimate=update(truth,torch.tensor([[.1,0.,0.]]),torch.zeros(1,3),torch.ones(1))[0]
    s=Scene(estimate,torch.zeros(24),.2,{'appearance':'asset'},torch.eye(3),{},torch.rand(1,3,8,8),torch.ones(1,1,8,8),torch.eye(3))
    result=paired_scenes([s],truth,lambda asset,pose,k,size:dict(pose=pose))[0]
    assert result.rgb is s.rgb and result.depth is s.depth and result.affine is s.affine and result.k_crop is s.k_crop
    assert not torch.equal(result.pose,s.pose)
    torch.testing.assert_close(result.state[:6],result.pose[:3,:2].T.flatten())
