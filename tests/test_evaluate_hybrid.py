import json
import numpy as np
import pytest
import torch
from torch import nn
import lip.evaluate as ev

@pytest.mark.parametrize('hybrid',[False,True])
def test_evaluation_commits_refined_state_without_future_gt(tmp_path,monkeypatch,hybrid):
    index=tmp_path/'index';index.mkdir()
    poses=np.tile(np.eye(4,dtype='f4'),(3,1,1));poses[:,2,3]=1.;poses[1:,0,3]=99.
    np.savez(index/'poses.npz',frames=np.arange(3),poses=poses)
    np.savez(index/'mesh.npz',center=np.zeros(3,dtype='f4'),diameter=np.float32(1),
             vertices=np.eye(3,dtype='f4')*.01,points=np.eye(3,dtype='f4')*.01)
    sid='subject/sequence/camera';s=dict(stream_id=sid,split='val',object_id=1,mesh_cache='mesh.npz',
        pose_cache='poses.npz',intrinsics=np.eye(3).tolist(),mesh_path='mesh.obj')
    (index/'streams.jsonl').write_text(json.dumps(s)+'\n')
    monkeypatch.setattr(ev,'check_data_gate',lambda *a:dict(complete=True,split_hash='s',mesh_hash='m',fps=30,depth_scale_to_m=.001))
    monkeypatch.setattr(ev,'motion_thresholds',lambda *a:dict(center_d_per_sec=1,rotation_rad_per_sec=1))
    monkeypatch.setattr(ev,'Renderer',lambda *a:object())
    monkeypatch.setattr(ev,'visibility',lambda *a:.8)
    monkeypatch.setattr(ev,'overlay',lambda *a:None)
    monkeypatch.setattr(ev,'read_frame',lambda root,s,f,scale:(np.full((3,8,8),f/10,dtype='f4'),np.ones((1,8,8),dtype='f4')))
    seen=[]
    def features(rgb,depth,history,times,valid,pv,base,*a):
        assert not pv[-1]
        seen.append((base.clone(),history.clone()))
        return dict(T_base_centered=base),{}
    monkeypatch.setattr(ev,'build_features',features)
    class Model(nn.Module):
        def __init__(self):super().__init__();self.p=nn.Parameter(torch.zeros(()))
        def forward(self,T_base_centered):
            pose=T_base_centered.clone();pose[:,0,3]+=.01
            return dict(pose_centered=pose)
    calls=[]
    def fp(pose,rgb,depth,k,path,center):
        assert not torch.is_grad_enabled() and not pose.requires_grad
        calls.append(float(rgb[0,0,0]));p=pose.clone();p[0,3]+=.1;return p
    c=dict(seed=42,precision='fp32',clip_length=2,image_size=8,crop_expansion=2.)
    out=tmp_path/'eval'
    ev.evaluate(Model(),c,tmp_path,index,out,fp_transition=fp if hybrid else None,stream_ids=[sid])
    rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
    assert len(rows)==3
    expected=[0,.11,.22] if hybrid else [0,.01,.02]
    assert [r['pose_centered'][0][3] for r in rows]==pytest.approx(expected)
    assert float(seen[1][0][0,3])==pytest.approx(expected[1])
    assert float(seen[1][1][-2,0,3])==pytest.approx(expected[1])
    assert calls==pytest.approx([.1,.2] if hybrid else [])
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['completed']
    assert manifest['history_state_source']==('post_FP' if hybrid else 'LIP')
