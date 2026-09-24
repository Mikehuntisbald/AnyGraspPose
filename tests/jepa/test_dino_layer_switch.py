from types import SimpleNamespace
import torch
from torch import nn
import pytest
from lip.jepa.encoder import FrozenDINO
from lip.unified.losses import reconstruction_loss,DEFAULT_WEIGHTS
from lip.unified.recovery_focus import focus_loss


def test_one_based_layer_selection_and_online_gradient():
    class Backbone(nn.Module):
        def get_intermediate_layers(self,rgb,n,reshape,norm):
            assert n==[3,10] and not reshape and norm
            x=rgb.mean().expand(len(rgb),256,384)
            return x+3,x+10
    encoder=FrozenDINO.__new__(FrozenDINO);nn.Module.__init__(encoder)
    encoder.backbone=Backbone();encoder.set_feature_layers([4,11]);encoder.trainable_encoder=True
    rgb=torch.randn(1,3,8,8,requires_grad=True);a,b=encoder(rgb)
    assert torch.allclose(b-a,torch.full_like(a,7))
    (a.sum()+b.sum()).backward();assert rgb.grad.abs().sum()>0
    for layers in ([0,11],[4,13],[11,4],[4,4],[4]):
        with pytest.raises(ValueError):encoder.set_feature_layers(layers)


def fixture():
    torch.manual_seed(9)
    f=torch.randn(1,4,8);t=torch.randn(1,4,8)
    out=dict(f_predicted=f.clone().requires_grad_(),f_mid_predicted=f.clone().requires_grad_(),
        surface_xyz=torch.zeros(1,3,2,2),surface_depth_residual=torch.zeros(1,1,2,2),
        geometry_valid_logits=torch.zeros(1,1,2,2),evidence_logits=torch.zeros(1,4),support_logits=torch.zeros(1,4),log_feature_error=torch.zeros(1,4))
    w=torch.ones(1,4);dense=torch.ones(1,1,2,2,dtype=torch.bool)
    teacher=SimpleNamespace(real_mid=t,real_last=t,proxy_mid=t,proxy_last=t,hidden_real_weight=w,visible_weight=w*0,
        proxy_weight=w,proxy_visible_weight=w*0,proxy_hidden_weight=w, surface_xyz=out['surface_xyz'],surface_depth_residual=out['surface_depth_residual'],
        geometry_weight=dense,geometry_visible_weight=dense,geometry_hidden_weight=~dense,geometry_real_weight=dense,geometry_proxy_weight=~dense,
        geometry_valid_label=dense.float(),visible_label=w,support_label=w,camera_rotation=torch.eye(3)[None],camera_translation_d=torch.zeros(1,3),
        camera_rays=torch.zeros(1,3,2,2),base_depth_d=torch.zeros(1))
    return out,teacher


def test_reconstruction_equal_weights_have_equal_head_gradients_and_legacy_is_four_to_one():
    for wm,wl in ((.625,.625),(.25,1.)):
        out,t=fixture();weights={k:0. for k in DEFAULT_WEIGHTS};weights.update(real_feature=1.,cad_feature=.5,feature_mid_weight=wm,feature_last_weight=wl)
        loss,_=reconstruction_loss(out,t,weights);loss.backward()
        assert torch.allclose(out['f_mid_predicted'].grad/wm,out['f_predicted'].grad/wl,atol=1e-6)


def test_spatial_losses_respect_the_same_equal_layer_weights():
    out,t=fixture();weights=dict(cad_feature=.5,spatial_centered=.25,spatial_correspondence=.05,camera_consistency=0.,feature_mid_weight=.625,feature_last_weight=.625)
    loss,_=focus_loss(out,t,weights);loss.backward()
    assert torch.allclose(out['f_mid_predicted'].grad,out['f_predicted'].grad,atol=1e-6)
