"""Test the real unroll runner's loss/feedback boundary with cheap pose dynamics."""
import pytest
import torch
from torch import nn
from lip.engine.stream_training import StreamTrainingModule,supervision_positions
from lip.geometry.so3 import update


class Cache:
    layers=((),(),(),())
    kv_bytes=0
    def detach(self):return self


class FrameActions(nn.Module):
    def __init__(self):
        super().__init__();self.actions=nn.Parameter(torch.zeros(56,6));self.grad_enabled=[]
    def forward(self,features,meta,cache):
        self.grad_enabled.append(torch.is_grad_enabled());delta=self.actions[meta.frame_id-1]
        pose=update(features['T_base_centered'],delta[:,:3],delta[:,3:],features['object_diameter_m'])
        return dict(pose_centered=pose),Cache()


def execute(monkeypatch,burn,startup=0):
    def build(rgb,depth,base,k,mesh,renderer,*args):
        return dict(T_base_centered=base,object_diameter_m=mesh['diameter']),dict(role_bias=torch.zeros(17))
    monkeypatch.setattr('lip.engine.stream_training.build_current_features',build)
    pose=torch.eye(4);pose[2,3]=.6;target=pose.repeat(56,1,1);target[:,0,3]=.01
    sample=dict(initial_pose=pose,targets=target,timestamps=torch.arange(57,dtype=torch.float64)/30,frames=torch.arange(57),
        rgb=torch.zeros(56,3,2,2),depth=torch.ones(56,1,2,2),k=torch.eye(3),sample=dict(seed=42),
        mesh=dict(diameter=torch.tensor(.1),points=torch.zeros(8,3)))
    config=dict(burn_in_frames=burn,supervised_unroll_frames=48,startup_supervision_frames=startup,initial_pose_noise=False,precision='fp32',image_size=2,crop_expansion=2.)
    tracker=FrameActions();result=StreamTrainingModule(tracker,config,None)([sample],True);result['loss'].backward()
    return tracker,result


def test_existing_burnin_excludes_first_eight_actions_from_direct_loss(monkeypatch):
    tracker,result=execute(monkeypatch,8)
    assert tracker.grad_enabled==[False]*8+[True]*48
    assert not tracker.actions.grad[:8].count_nonzero()
    assert (tracker.actions.grad[8:,3].abs()>0).all()
    assert result['supervised_frames']==48 and result['predictions'].shape[1]==56


def test_zero_burnin_supervises_first_update_without_extra_target_budget(monkeypatch):
    tracker,result=execute(monkeypatch,0)
    assert tracker.grad_enabled==[True]*48
    assert (tracker.actions.grad[:48,3].abs()>0).all()
    assert not tracker.actions.grad[48:].count_nonzero()
    assert result['supervised_frames']==48 and result['predictions'].shape[1]==48


def test_startup_supervision_keeps_56_observations_and_48_selected_targets(monkeypatch):
    tracker,result=execute(monkeypatch,8,8)
    selected=supervision_positions(8,48,8)
    assert len(selected)==56 and sum(selected)==48 and all(selected[:8])
    assert [i for i in range(8,56) if not selected[i]]==[13,19,25,31,37,43,49,55]
    assert tracker.grad_enabled==[True]*56
    assert all(bool(tracker.actions.grad[i,3].abs()>0)==flag for i,flag in enumerate(selected))
    assert result['startup_supervised_frames']==result['omitted_late_targets']==8
    assert result['supervised_frames']==48 and result['predictions'].shape[1]==56
    with pytest.raises(ValueError):supervision_positions(0,48,8)
    with pytest.raises(ValueError):supervision_positions(8,16,8)
