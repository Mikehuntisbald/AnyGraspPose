"""Full-resolution CAD appearance with mipmapped minification for RGB-D targets."""
import torch
from lip.geometry.appearance_renderer import AppearanceRenderer


class FullTextureRenderer(AppearanceRenderer):
    @torch.no_grad()
    def __call__(self, mesh, pose, k, size=224):
        with torch.autocast(self.device.type, enabled=False):
            v=mesh['vertices'].float();f=mesh['faces'].int();pose=pose.float();k=k.float()
            camera=v@pose[:3,:3].T+pose[:3,3];z=camera[:,2];pixel=camera@k.T;near,far=.001,100.
            clip=torch.stack((2*(pixel[:,0]+.5*z)/size-z,2*(pixel[:,1]+.5*z)/size-z,
                (far+near)/(far-near)*z-2*far*near/(far-near),z),-1)
            rast,db=self.dr.rasterize(self.context,clip[None].contiguous(),f,resolution=[size,size])
            attributes=torch.cat((camera[:,2:],v,mesh['uv'].float(),mesh['normals'].float()@pose[:3,:3].T),-1)[None]
            value,uv_da=self.dr.interpolate(attributes.contiguous(),rast,f,rast_db=db,diff_attrs=[4,5])
            texture=mesh['texture'].float().contiguous()
            key=(texture.data_ptr(),texture._version)
            if mesh.get('_mip_identity')!=key:
                mesh['_mip']=self.dr.texture_construct_mip(texture);mesh['_mip_identity']=key
            color=self.dr.texture(texture,value[...,4:6].contiguous(),uv_da=uv_da.contiguous(),mip=mesh['_mip'],
                filter_mode='linear-mipmap-linear',boundary_mode='clamp')[0]
            normal=torch.nn.functional.normalize(value[0,...,6:9],dim=-1)
            color=color*(.8+.2*normal[...,2:].abs());mask=rast[0,:,:,3]>0
            rgb=torch.where(mask[:,:,None],color,torch.full_like(color,.5)).permute(2,0,1)
            geometry=torch.where(mask[:,:,None],value[0,:,:,:4],0).permute(2,0,1)
            return dict(rgb=rgb,depth=geometry[:1],xyz=geometry[1:],mask=mask)
