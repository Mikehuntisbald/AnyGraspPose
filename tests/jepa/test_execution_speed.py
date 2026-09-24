import torch
from lip.geometry.crop import crop_matrix,crop_images
from lip.unified.features import camera_points
from lip.unified.execution_speed import crop_matrix_fast,crop_images_fast,camera_points_fast,FrozenFeatureGraph


def test_crop_and_rays_match_reference_with_front_and_behind_vertices():
    torch.manual_seed(17)
    k=torch.tensor([[400.,0,112],[0,400,112],[0,0,1.]])
    base=torch.eye(4);base[2,3]=1
    rgb=torch.rand(1,3,224,224);depth=torch.rand(1,1,224,224)
    for shift in (0.,-1.,-3.):
        vertices=torch.randn(40,3)*.1;vertices[:,2]+=shift
        a,ka=crop_matrix(vertices,base,k);b,kb=crop_matrix_fast(vertices,base,k)
        assert torch.equal(a,b) and torch.equal(ka,kb)
        for mode in ('bilinear','nearest'):
            assert torch.equal(crop_images(rgb,a,mode=mode),crop_images_fast(rgb,b,mode=mode))
        assert torch.equal(camera_points(depth,ka),camera_points_fast(depth,kb))


def test_graph_rejects_trainable_encoder():
    import pytest
    with pytest.raises(ValueError):FrozenFeatureGraph(torch.nn.Linear(3,3))


def test_vector_teacher_preserves_masks_geometry_and_rgb():
    from types import SimpleNamespace
    from lip.unified.features import build_teachers
    from lip.unified.fast_teacher import build_fast_teacher
    torch.manual_seed(18);size=224
    k=torch.tensor([[400.,0,112],[0,400,112],[0,0,1.]])
    pose=torch.eye(4);pose[2,3]=1
    silhouette=torch.zeros(size,size,dtype=torch.bool);silhouette[16:-16,16:-16]=True
    rendering=dict(rgb=torch.rand(3,size,size),depth=torch.ones(1,size,size)*silhouette,
                   xyz=torch.rand(3,size,size)*.1,mask=silhouette)
    class Renderer:
        def __call__(self,*args):return rendering
        def render_many(self,meshes,*args):return [rendering]*len(meshes)
    class Encoder:
        def __call__(self,rgb):
            a=torch.nn.functional.avg_pool2d(rgb,14,14).flatten(2).transpose(1,2)
            return a,a*.5
    scenes=[];visible=[];added=[]
    for i in range(3):
        depth=torch.ones(1,1,size,size);depth[:,:,50:70]=0
        scenes.append(SimpleNamespace(rgb=torch.rand(1,3,size,size),depth=depth,pose=pose,k_crop=k,
            affine=torch.eye(3),bounds=torch.ones_like(depth,dtype=torch.bool),diameter=.2,cad={'appearance':None}))
        v=silhouette[None].clone();v[:,:,:100]=False;visible.append(v)
        mask=torch.zeros_like(depth,dtype=torch.bool);mask[:,:,:,80:160]=i>0;added.append(mask)
    args=(Encoder(),scenes,pose[None].expand(3,-1,-1),torch.stack(visible),added,Renderer())
    slow=build_teachers(*args,real_geometry_max_radius_d=1.)
    fast=build_fast_teacher(*args,max_radius_d=1.,batch_render=True)
    for name in slow.__dataclass_fields__:
        a,b=getattr(slow,name),getattr(fast,name)
        if isinstance(a,torch.Tensor):assert torch.allclose(a,b,rtol=0,atol=0,equal_nan=True),name


def test_batched_geometry_and_static_mean_cache_match_original():
    from types import SimpleNamespace
    from lip.geometry.crop import geometry_channels
    from lip.unified.execution_speed import geometry_inputs_fast
    import torch.nn.functional as F
    torch.manual_seed(19);b=3;size=224
    k=torch.tensor([[400.,0,112],[0,400,112],[0,0,1.]])
    base=torch.eye(4)[None].repeat(b,1,1);base[:,2,3]=1
    diameter=torch.tensor([.2,.3,.4]);enabled=torch.tensor([True,False,True])
    depth=torch.rand(b,1,size,size);depth[:,:,16:32]=0;bounds=torch.ones_like(depth,dtype=torch.bool)
    scenes=[SimpleNamespace(k_crop=k,render=dict(depth=torch.rand(1,size,size),xyz=torch.rand(3,size,size)),cad={'features':torch.rand(30,12)}) for i in range(b)]
    geo=[];xyz=[];valid=[];cad=[]
    for i,s in enumerate(scenes):
        g=geometry_channels(depth[i:i+1],s.render['depth'][None],s.render['xyz'][None],diameter[i],base[i,2,3]);g[:,2:]*=enabled[i];geo.append(g)
        camera=camera_points(depth[i:i+1],k);local=(camera-base[i,:3,3])@base[i,:3,:3]/diameter[i]
        m=depth[i:i+1]>0;mass=F.avg_pool2d(m.float(),14,14)
        xyz.append((F.avg_pool2d(local.permute(2,0,1)[None]*m,14,14)/mass.clamp_min(1e-6)).flatten(2).transpose(1,2)[0])
        valid.append(mass.flatten()>0);cad.append(s.cad['features'].mean(0))
    actual=geometry_inputs_fast(scenes,depth,bounds,base,diameter,enabled)
    for a,v in zip(actual,(torch.cat(geo),torch.stack(xyz),torch.stack(valid),torch.stack(cad))):assert torch.equal(a,v)
    before=actual[-1].clone();scenes[0].cad['features'].add_(1)
    after=geometry_inputs_fast(scenes,depth,bounds,base,diameter,enabled)[-1]
    assert torch.allclose(after[0]-before[0],torch.ones_like(after[0])) and torch.equal(after[1:],before[1:])
