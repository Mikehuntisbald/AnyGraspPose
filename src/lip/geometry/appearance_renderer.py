"""Textured CAD RGB/depth/XYZ in LIP's centered-meter/OpenCV convention.

Independent nvdiffrast renderer; no FoundationPose import or runtime resource.
The object texture is read once and retained with the mesh on the GPU.
"""
from pathlib import Path
import numpy as np
import torch
from lip.geometry.renderer import Renderer


def load_appearance_mesh(path,center,device='cuda',texture_size=1024):
    import trimesh
    from PIL import Image
    mesh=trimesh.load(str(Path(path)),process=False,force='mesh')
    if mesh.visual.kind!='texture' or getattr(mesh.visual.material,'image',None) is None:
        raise ValueError('An explicit textured CAD model is required')
    texture=mesh.visual.material.image.convert('RGB')
    texture.thumbnail((texture_size,texture_size),Image.Resampling.LANCZOS)
    uv=np.asarray(mesh.visual.uv,dtype='f4').copy();uv[:,1]=1-uv[:,1]
    if len(uv)!=len(mesh.vertices):raise ValueError('Texture UV/vertex topology mismatch')
    return dict(vertices=torch.as_tensor(np.asarray(mesh.vertices,dtype='f4')-np.asarray(center,dtype='f4'),device=device),
        faces=torch.as_tensor(np.asarray(mesh.faces,dtype='i4'),device=device),uv=torch.as_tensor(uv,device=device),
        texture=torch.as_tensor(np.array(texture,dtype='f4')/255,device=device)[None],
        normals=torch.as_tensor(np.asarray(mesh.vertex_normals,dtype='f4').copy(),device=device))


class AppearanceRenderer(Renderer):
    def __init__(self,device='cuda'):
        super().__init__(device,backend='cuda')

    def geometry(self,mesh,pose,k,size=224):
        return super().__call__(mesh,pose,k,size)

    @torch.no_grad()
    def __call__(self,mesh,pose,k,size=224):
        with torch.autocast(self.device.type,enabled=False):
            v=mesh['vertices'].float();f=mesh['faces'].int();pose=pose.float();k=k.float()
            camera=v@pose[:3,:3].T+pose[:3,3];z=camera[:,2];pixel=camera@k.T;near,far=.001,100.
            clip=torch.stack((2*(pixel[:,0]+.5*z)/size-z,2*(pixel[:,1]+.5*z)/size-z,
                (far+near)/(far-near)*z-2*far*near/(far-near),z),-1)
            rast,_=self.dr.rasterize(self.context,clip[None].contiguous(),f,resolution=[size,size])
            attributes=torch.cat((camera[:,2:],v,mesh['uv'].float(),mesh['normals'].float()@pose[:3,:3].T),-1)[None]
            value,_=self.dr.interpolate(attributes.contiguous(),rast,f);mask=rast[0,:,:,3]>0
            color=self.dr.texture(mesh['texture'].float().contiguous(),value[...,4:6].contiguous(),filter_mode='linear',boundary_mode='clamp')[0]
            normal=value[0,...,6:9];normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            # Fixed mild headlight plus ambient light; explicit gray background.
            color=color*(.8+.2*normal[...,2:].abs())
            rgb=torch.where(mask[:,:,None],color,torch.full_like(color,.5)).permute(2,0,1)
            geometry=torch.where(mask[:,:,None],value[0,:,:,:4],0).permute(2,0,1)
            return dict(rgb=rgb,depth=geometry[:1],xyz=geometry[1:],mask=mask)
