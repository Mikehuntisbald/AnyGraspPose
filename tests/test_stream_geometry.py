import numpy as np
import pytest
import torch
from lip.engine.stream_features import build_current_features
from lip.geometry.renderer import Renderer
from lip.geometry.crop import project
from lip.geometry.so3 import center_pose,original_pose,exp


def fixture():
    import trimesh
    m=trimesh.creation.box(extents=[.1,.08,.06])
    mesh=dict(vertices=np.asarray(m.vertices,dtype='f4'),faces=np.asarray(m.faces,dtype='i4'),
              points=np.tile(np.asarray(m.vertices,dtype='f4'),(64,1)),center=np.array([.03,-.02,.01],dtype='f4'),diameter=np.float32(.15))
    t=torch.eye(4);t[2,3]=.6;k=torch.tensor([[100.,0,31.5],[0,100.,31.5],[0,0,1.]])
    return mesh,t,k


def test_current_geometry_coordinates_state24_and_one_render():
    mesh,base,k=fixture();real=Renderer('cpu');calls=[]
    def render(*a):calls.append(1);return real(*a)
    f,d=build_current_features(torch.rand(3,64,64),torch.ones(1,64,64)*.6,base,k,mesh,render,
        1789000000.034,1789000000.,size=32)
    assert f['rgb'].shape==(3,32,32) and f['geometry'].shape==(9,32,32)
    assert f['state_input'].shape==(24,) and len(calls)==1
    assert f['state_input'][20]==0 and f['state_input'][22]==0
    assert abs(f['state_input'][21].item()-.034)<1e-6
    camera=torch.tensor(mesh['vertices'])+base[:3,3]
    uv=project(camera,k);mapped=torch.cat((uv,torch.ones(len(uv),1)),-1)@d['A'].T
    torch.testing.assert_close(mapped[:,:2],project(camera,d['K_crop']),atol=3e-5,rtol=1e-5)
    restored=torch.cat((f['source_xy']*64,torch.ones(16,1)),-1)@d['A'].T
    centers=(torch.arange(4)+.5)*8-.5
    torch.testing.assert_close(restored[:4,0],centers)
    c=torch.tensor(mesh['center']);torch.testing.assert_close(original_pose(center_pose(base,c),c),base)


def test_motion_state_comes_from_accepted_history_and_geometry_fallback():
    mesh,base,k=fixture();previous=base.clone();previous[:3,:3]=exp(torch.tensor([0.,0.,-.1]));previous[0,3]-=.015
    f,d=build_current_features(torch.zeros(3,64,64),torch.full((1,64,64),float('nan')),base,k,mesh,Renderer('cpu'),
        .15,.1,previous,.05,size=32)
    assert torch.isfinite(f['geometry']).all() and d['depth_had_nonfinite']
    torch.testing.assert_close(f['state_input'][14:17],torch.tensor([0.,0.,.1]),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(f['state_input'][17:20],torch.tensor([.1,0,0]),atol=1e-5,rtol=1e-5)
    assert f['state_input'][22]==1 and f['state_input'][23]==0
    base[2,3]=-1
    f,d=build_current_features(torch.zeros(3,64,64),torch.zeros(1,64,64),base,k,mesh,Renderer('cpu'),.2,.1,size=32)
    assert not d['geometry_reliable'] and torch.equal(d['role_bias'],torch.zeros(17))
