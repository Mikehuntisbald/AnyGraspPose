"""Explicit synthetic correctness fixture, never an alternative DexYCB dataset."""
import numpy as np
import torch
import trimesh
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import exp

def synthetic_item(length=8,size=640):
 m=trimesh.creation.box(extents=[.1,.08,.06]);v=np.asarray(m.vertices,dtype='f4')
 mesh=dict(vertices=v,faces=np.asarray(m.faces,dtype='i4'),center=np.array([.03,-.02,.01],dtype='f4'),diameter=np.float32(np.linalg.norm([.1,.08,.06])),points=np.tile(v,(64,1)))
 k=torch.tensor([[600.,0,(size-1)/2],[0,600.,(min(480,size)-1)/2],[0,0,1.]])
 renderer=Renderer('cpu');poses=[];rgb=[];depth=[]
 for i in range(length+4):
  t=torch.eye(4);t[:3,:3]=exp(torch.tensor([0.,.005*i,.01*i]));t[:3,3]=torch.tensor([.001*i,0,.6]);poses.append(t)
  if i:
   d,xyz=renderer(mesh,t,k,size);d=d[:,:480];xyz=xyz[:,:480]
   rgb.append((xyz/.15+.5).clamp(0,1));depth.append(d)
 return dict(rgb=torch.stack(rgb),depth=torch.stack(depth),poses=torch.stack(poses),frames=torch.arange(length+4),times=torch.arange(1,length+4)/30,
             k=k,mesh=mesh,effective=length,seed=42,stream=dict(stream_id='SYNTHETIC',object_id=1),sample=dict(source='synthetic_correctness_fixture'))
