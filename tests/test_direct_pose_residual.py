import pytest
import torch
from lip.models.direct_pose_residual import DirectPoseResidual,apply_direct_residual
from lip.geometry.so3 import update,original_pose


def fixture():
    base=torch.eye(4)[None].repeat(2,1,1);base[:,:3,3]=torch.tensor([.3,-.2,1.])
    d=torch.tensor([.1,.4]);center=torch.tensor([[.02,.01,0.],[.01,0.,.03]])
    delta=torch.tensor([[.03,-.01,.02,.01,-.02,.03],[-.05,.01,0.,.02,0.,-.01]])
    pose=update(base,delta[:,:3],delta[:,3:],d)
    parent=dict(delta_rotvec=delta[:,:3],delta_center_norm=delta[:,3:],pose_centered=pose,pose_original=original_pose(pose,center))
    return parent,dict(T_base_centered=base,object_diameter_m=d,mesh_center=center)


def test_zero_head_preserves_parent_pose_and_action_exactly():
    head=DirectPoseResidual();parent,features=fixture()
    residual=head(torch.randn(2,256));assert torch.count_nonzero(residual)==0
    out=apply_direct_residual(parent,features,residual)
    for key in parent:assert torch.equal(parent[key],out[key])


def test_camera_axis_rotation_does_not_rotate_center_translation():
    parent,features=fixture();residual=torch.zeros(2,6);residual[:,2]=.2
    out=apply_direct_residual(parent,features,residual)
    assert torch.equal(out['pose_centered'][:,:3,3],parent['pose_centered'][:,:3,3])
    residual.zero_();residual[:,3]=.1;out=apply_direct_residual(parent,features,residual)
    torch.testing.assert_close(out['pose_centered'][:,0,3]-parent['pose_centered'][:,0,3],features['object_diameter_m']*.1)
    assert torch.equal(out['pose_centered'][:,:3,:3],parent['pose_centered'][:,:3,:3])


def test_both_zero_output_branches_receive_finite_nonzero_gradient():
    head=DirectPoseResidual();parent,features=fixture()
    out=apply_direct_residual(parent,features,head(torch.randn(2,256)))
    loss=out['pose_centered'][:,0,1].sum()+out['pose_centered'][:,0,3].sum();loss.backward()
    for branch in (head.rotation,head.center):
        assert torch.isfinite(branch[-1].weight.grad).all() and branch[-1].weight.grad.abs().sum()>0


def test_invalid_latent_or_action_shape_rejected():
    head=DirectPoseResidual()
    with pytest.raises(ValueError):head(torch.zeros(1,1,256))
    parent,features=fixture()
    with pytest.raises(ValueError):apply_direct_residual(parent,features,torch.zeros(2,3))
