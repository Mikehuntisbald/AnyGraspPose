from types import SimpleNamespace
import torch
from lip.unified.pose_geometry import objective,METRICS


def example():
    torch.manual_seed(1);b,h,w,n,k=1,4,4,2,3
    leaf=lambda *shape:torch.randn(*shape,requires_grad=True)
    mask=torch.ones(b,1,h,w,dtype=torch.bool)
    target=SimpleNamespace(surface_xyz=torch.zeros(b,3,h,w),surface_depth_residual=torch.ones(b,1,h,w)*.1,
        geometry_real_weight=mask,geometry_proxy_weight=mask&False,geometry_valid_label=mask.float(),
        camera_rotation=torch.eye(3)[None],camera_translation_d=torch.zeros(b,3),camera_rays=torch.ones(b,3,h,w),base_depth_d=torch.ones(b),
        visible_label=torch.ones(b,n),support_label=torch.ones(b,n),
        cad_surface_target_real=torch.tensor([[[1.,0,0,0],[0,1.,0,0]]]),cad_surface_target_proxy=torch.zeros(b,n,k+1),
        cad_surface_weight_real=torch.ones(b,n),cad_surface_weight_proxy=torch.zeros(b,n))
    output=dict(pose_centered=torch.eye(4)[None].clone().requires_grad_(),surface_xyz=leaf(b,3,h,w),surface_depth_residual=leaf(b,1,h,w),
        geometry_valid_logits=leaf(b,1,h,w),coarse_surface=leaf(b,5,h,w),evidence_logits=leaf(b,n),support_logits=leaf(b,n),
        cad_match_log_prob=leaf(b,n,k+1).log_softmax(-1),cad_surface_available=torch.ones(b,k,dtype=torch.bool),
        rope_patch_valid=torch.ones(b,n,dtype=torch.bool),rope_measured_mask=torch.ones(b,n,dtype=torch.bool),
        rope_recovered_mask=torch.zeros(b,n,dtype=torch.bool),rope_fallback_mask=torch.zeros(b,n,dtype=torch.bool),
        rope_measured_trust=torch.ones(b,n),rope_recovered_trust=torch.zeros(b,n),
        f_predicted=leaf(b,n,8),f_mid_predicted=leaf(b,n,8),log_feature_error=leaf(b,n))
    weights=dict(pose=1.,geometry_beta=.05,geometry_proxy_factor=.5,surface_xyz=.25,surface_depth=.25,camera_consistency=.1,geometry_validity=.1,visibility_support=.1,cad_surface_correspondence=.1,coarse_surface=.1)
    truth=torch.eye(4)[None];truth[:,2,3]=.1
    return output,target,truth,torch.randn(b,10,3),torch.ones(b),weights


def test_no_feature_targets_or_gradients():
    args=example();loss,stats=objective(*args);loss.backward();o=args[0]
    assert torch.isfinite(loss) and stats.numel()==len(METRICS)
    assert all(o[k].grad is None for k in ('f_predicted','f_mid_predicted','log_feature_error'))
    assert all(o[k].grad.abs().sum()>0 for k in ('pose_centered','surface_xyz','surface_depth_residual','coarse_surface'))


def test_feature_outputs_cannot_change_loss():
    args=example();before,_=objective(*args)
    for k in ('f_predicted','f_mid_predicted','log_feature_error'):args[0][k]=torch.full_like(args[0][k],float('nan'))
    after,_=objective(*args)
    torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_compiled_loss_matches_eager():
    args=example();eager,stats=objective(*args)
    compiled=torch.compile(objective,backend='aot_eager',fullgraph=True)
    loss,other=compiled(*args)
    torch.testing.assert_close(loss,eager);torch.testing.assert_close(other,stats)
