import torch
from lip.models.basin import BasinCritic,basin_loss,candidate_poses
from lip.geometry.so3 import update

def test_candidate_deltas_reconstruct_update_and_original():
 base=torch.eye(4);base[2,3]=.7;pred=update(base,torch.tensor([.1,-.03,.02]),torch.tensor([.02,-.04,.01]),torch.tensor(.2))
 poses,deltas=candidate_poses(base,pred,.2,torch.Generator().manual_seed(731))
 assert poses.shape==(6,4,4) and torch.equal(poses[0],pred)
 actual=update(base.expand(6,-1,-1),deltas[:,:3],deltas[:,3:],torch.full((6,),.2))
 assert torch.allclose(actual,poses,atol=2e-6)
 assert torch.allclose((poses[1,:3,3]-pred[:3,3]).norm(),torch.tensor(.004),atol=1e-7)
 assert torch.allclose(poses[1,:3,3]+poses[2,:3,3],2*pred[:3,3],atol=1e-7)

def test_frozen_critic_preserves_action_gradient_and_detaches_scene():
 torch.manual_seed(731);q=BasinCritic().eval().requires_grad_(False)
 z=torch.randn(4,256,requires_grad=True);d=torch.randn(4,6,requires_grad=True)
 loss=basin_loss(q,z,d);loss.backward()
 assert z.grad is None and d.grad is not None and torch.isfinite(d.grad).all() and d.grad.abs().sum()>0
 assert all(p.grad is None for p in q.parameters())

def test_candidate_conditioning_and_calibration_temperature():
 torch.manual_seed(731);q=BasinCritic().eval();z=torch.randn(2,256);z[1]=z[0];d=torch.zeros(2,6);d[1,0]=.2
 logits=q(z,d);assert not torch.equal(logits[0],logits[1]);q.temperature.fill_(2)
 assert torch.allclose(q(z,d),logits/2)

def test_integrated_basin_loss_keeps_pose_supervision():
    from lip.models.tracker import Tracker
    from lip.data.synthetic import synthetic_item
    from lip.geometry.renderer import Renderer
    from lip.engine.runtime import batch_step
    torch.set_num_threads(2);torch.manual_seed(731)
    actor=Tracker(False,dropout=0.).eval();q=BasinCritic().eval().requires_grad_(False)
    item=synthetic_item(length=2,size=64);cfg=dict(clip_length=2,image_size=32,precision='fp32',crop_expansion=2.)
    values,_=batch_step(actor,[item],Renderer('cpu'),cfg,1,False,True,basin_critic=q,basin_weight=.01)
    assert abs(values['loss']-(values['pose_loss']+.01*values['basin_loss']))<1e-6
    assert actor.head[-1].weight.grad is not None and torch.isfinite(actor.head[-1].weight.grad).all()
    assert all(p.grad is None for p in q.parameters())
