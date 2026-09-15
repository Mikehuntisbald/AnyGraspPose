import hashlib
import json
import numpy as np
import pytest
import torch
from torch import nn
from lip.data.external_initializers import load_train_initializers,request_real_initializer
from lip.engine.stream_training import StreamTrainingModule
from lip.geometry.so3 import update


def test_train_initializer_provenance_rejects_val_gt_and_object_mismatch(tmp_path):
    T=np.eye(4);T[2,3]=.6;path=tmp_path/'initializers.json'
    streams=[dict(stream_id='a',split='train',num_frames=4,object_id=1)];audit=dict(split_hash='split',mesh_hash='mesh')
    value=dict(completed=True,split='train',uses_gt_pose=False,fp_calls=0,checkpoint_sha256='weight',inference_source_sha256='source',**audit,
        initializers={'a|0':dict(stream_id='a',frame_index=0,object_id=1,score=.5,pose_original=T.tolist()),'a|1':None})
    def load(v):
        path.write_text(json.dumps(v));return load_train_initializers(path,hashlib.sha256(path.read_bytes()).hexdigest(),streams,audit)
    assert load(value)['initializers']['a|1'] is None
    for extra in (dict(split='val'),dict(uses_gt_pose=True),dict(fp_calls=1),dict(mesh_hash='other')):
        with pytest.raises(ValueError):load(dict(value,**extra))
    for extra in (dict(object_id=2),dict(frame_index=1),dict(pose_original=[0,1,2])):
        with pytest.raises(ValueError):load(dict(value,initializers={'a|0':dict(value['initializers']['a|0'],**extra)}))
    with pytest.raises(ValueError):load_train_initializers(path,'bad-sha',streams,audit)


def test_mixture_assignment_is_reproducible_and_has_explicit_endpoints():
    seeds=list(range(100))
    first=[request_real_initializer(s,.5) for s in seeds]
    assert first==[request_real_initializer(s,.5) for s in seeds]
    assert any(first) and not all(first)
    assert not any(request_real_initializer(s,0) for s in seeds)
    assert all(request_real_initializer(s,1) for s in seeds)


class Cache:
    layers=((),(),(),());kv_bytes=0
    def detach(self):return self


class ToyTracker(nn.Module):
    architecture_id='stream_dual_cross_residual'
    def __init__(self):
        super().__init__();self.actions=nn.Parameter(torch.zeros(57,6));self.frames=[];self.bases=[]
        with torch.no_grad():self.actions[0,3]=1.
    def forward(self,features,meta,cache):
        self.frames.append(int(meta.frame_id[0]));self.bases.append(features['T_base_centered'].detach().clone())
        delta=self.actions[meta.frame_id-1]
        return dict(pose_centered=update(features['T_base_centered'],delta[:,:3],delta[:,3:],features['object_diameter_m'])),Cache()


def test_priming_keeps_real_pose_does_not_commit_warm_prediction_and_preserves_budget(monkeypatch):
    def features(rgb,depth,base,k,mesh,renderer,*args):
        return dict(T_base_centered=base,object_diameter_m=mesh['diameter']),dict(role_bias=torch.zeros(17))
    monkeypatch.setattr('lip.engine.stream_training.build_current_features',features)
    T=torch.eye(4);T[2,3]=.6;real=T.clone();real[0,3]=.04;targets=T.repeat(56,1,1);targets[:,0,3]=.01
    s=dict(initial_pose=T,real_initial_pose=real,real_initialization_requested=True,real_initialization_missing=False,
        initial_rgb=torch.zeros(3,2,2),initial_depth=torch.ones(1,2,2),rgb=torch.zeros(56,3,2,2),depth=torch.ones(56,1,2,2),
        targets=targets,timestamps=torch.arange(7,64,dtype=torch.float64)/30,frames=torch.arange(7,64),k=torch.eye(3),sample=dict(seed=42),
        mesh=dict(diameter=torch.tensor(.1),points=torch.zeros(8,3)))
    c=dict(burn_in_frames=8,supervised_unroll_frames=48,startup_supervision_frames=8,initial_pose_noise=False,precision='fp32',
        image_size=2,crop_expansion=2.,time_unit=1/30,prime_initial_observation=True,real_initialization_probability=1.)
    model=ToyTracker();result=StreamTrainingModule(model,c,None)([s],True);result['loss'].backward()
    assert model.frames==list(range(1,58))
    assert torch.equal(model.bases[0][0],real) and torch.equal(model.bases[1][0],real)
    assert result['predictions'].shape[1]==56 and result['supervised_frames']==48
    assert result['primed_observations']==result['real_initializations']==1
    assert not model.actions.grad[0].count_nonzero() and model.actions.grad[1,3].abs()>0
    assert result['startup_supervised_frames']==result['omitted_late_targets']==8
