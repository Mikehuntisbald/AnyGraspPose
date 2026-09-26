from types import SimpleNamespace
import numpy as np
import torch
from lip.unified.cad_image_correspondence import (CADImageReadout,image_grid,sample_points,
    image_correspondence_targets,image_correspondence_loss,solve_correspondences)


def test_inverse_retrieval_returns_image_endpoint_and_shared_feature_gradients():
    torch.manual_seed(42)
    q=torch.eye(16)[None].requires_grad_()
    k=torch.eye(16)[None].requires_grad_()
    atlas=dict(atlas_query=q,atlas_keys=k,atlas_xyz=torch.zeros(1,16,3),atlas_available=torch.ones(1,16,dtype=torch.bool),atlas_temperature=.01)
    model=CADImageReadout(width=16,points=16,stride=1)
    out=model(atlas,(4,4))
    torch.testing.assert_close(out['cad_image_uv'],image_grid(4,4,1,'cpu')[None],atol=1e-4,rtol=0)
    labels=dict(uv=image_grid(4,4,1,'cpu').flip(0)[None],support=torch.ones(1,16,dtype=torch.bool),visible=torch.ones(1,16,dtype=torch.bool))
    labels.update(known_support=labels['support'],known_visible=labels['visible'],observed=labels['visible'],real=~labels['visible'],proxy=~labels['visible'])
    loss,_=image_correspondence_loss(out,labels);loss.backward()
    assert q.grad.norm()>0 and k.grad.norm()>0
    assert model.visibility[-1].weight.grad.norm()>0
    again=model(atlas,(4,4))
    assert torch.equal(out['cad_image_uv'],again['cad_image_uv'])


def test_projection_crop_centers_and_front_surface_not_backface_labels():
    n=16
    grid=image_grid(n,n,1,'cpu').reshape(n,n,2)
    k=torch.tensor([[[10.,0.,7.5],[0.,10.,7.5],[0.,0.,1.]]])
    xyz=torch.cat(((grid-7.5)/10,torch.zeros(n,n,1)),dim=-1).permute(2,0,1)[None]
    target=SimpleNamespace(camera_rotation=torch.eye(3)[None],camera_translation_d=torch.tensor([[0.,0.,1.]]),
        cad_geometry_valid=torch.ones(1,1,n,n,dtype=torch.bool),cad_geometry_xyz=xyz,cad_geometry_depth_m=torch.ones(1,1,n,n),
        geometry_real_weight=torch.zeros(1,1,n,n,dtype=torch.bool),geometry_proxy_weight=torch.zeros(1,1,n,n,dtype=torch.bool),real_geometry_eligible=None)
    # At center: front point, clear back point, outside point.
    output=dict(cad_image_xyz=torch.tensor([[[0.,0.,0.],[0.,0.,.2],[3.,0.,0.]]]),cad_image_available=torch.ones(1,3,dtype=torch.bool))
    labels=image_correspondence_targets(output,target,k,torch.ones(1),torch.ones(1,1,n,n,dtype=torch.bool))
    torch.testing.assert_close(labels['uv'][0,0],torch.tensor([7.5,7.5]))
    assert labels['support'].tolist()==[[True,False,False]]
    assert labels['visible'].tolist()==[[True,False,False]]
    assert labels['known_support'].all()
    target.geometry_real_weight[:]=True
    hidden=image_correspondence_targets(output,target,k,torch.ones(1),torch.zeros(1,1,n,n,dtype=torch.bool))
    assert hidden['real'].tolist()==[[True,False,False]]
    assert not hidden['visible'].any() and hidden['known_visible'][0,0]
    # Geometry is measured in meters; diameter changes must not change projection.
    target.cad_geometry_depth_m*=.2
    scaled=image_correspondence_targets(output,target,k,torch.tensor([.2]),torch.zeros(1,1,n,n,dtype=torch.bool))
    torch.testing.assert_close(scaled['uv'],hidden['uv'])
    assert torch.equal(scaled['support'],hidden['support'])


def test_sampling_pixel_centers_and_absent_cad():
    image=torch.arange(16).reshape(1,1,4,4).float()
    value=sample_points(image,image_grid(4,4,1,'cpu')[None])
    torch.testing.assert_close(value[0,:,0],image.flatten())
    torch.testing.assert_close(image_grid(2,2,2,'cpu'),torch.tensor([[.5,.5],[2.5,.5],[.5,2.5],[2.5,2.5]]))
    read=CADImageReadout(width=4,points=4,stride=1)
    out=read(dict(atlas_query=torch.zeros(1,16,4),atlas_keys=torch.zeros(1,4,4),atlas_xyz=torch.zeros(1,4,3),atlas_available=torch.zeros(1,4,dtype=torch.bool),atlas_temperature=.1),(4,4))
    assert torch.isfinite(out['cad_image_uv']).all() and not out['cad_image_confidence'].any()


def test_pnp_recovers_known_pose_with_outliers_and_has_safe_empty_fallback():
    import cv2
    rng=np.random.default_rng(42)
    xyz=rng.uniform(-.05,.05,(128,3))
    rotation=cv2.Rodrigues(np.array([.12,-.08,.15]))[0];t=np.array([.01,-.02,.5])
    k=np.array([[450.,0.,112.],[0.,450.,112.],[0.,0.,1.]])
    camera=xyz@rotation.T+t;uv=camera@k.T;uv=uv[:,:2]/uv[:,2:]
    uv[:32]=rng.uniform(0,224,(32,2))
    base=np.eye(4);base[2,3]=.55
    pose,receipt=solve_correspondences(xyz,uv,np.ones(128),k,base)
    assert receipt['accepted'] and receipt['inliers']>=90
    np.testing.assert_allclose(pose[:3,:3],rotation,atol=1e-5)
    np.testing.assert_allclose(pose[:3,3],t,atol=1e-5)
    fallback,receipt=solve_correspondences(xyz,uv,np.zeros(128),k,base)
    assert not receipt['accepted'];np.testing.assert_array_equal(fallback,base)


def test_balanced_visibility_cannot_reduce_positive_mass_by_adding_negatives():
    from lip.unified.cad_image_correspondence import balanced_binary_loss
    values=[]
    for negatives in (1,100):
        logits=torch.full((1,negatives+1),-4.,requires_grad=True)
        labels=torch.zeros_like(logits,dtype=torch.bool);labels[:,0]=True
        loss=balanced_binary_loss(logits,labels,torch.ones_like(labels))
        loss.backward();values.append((loss.detach(),logits.grad[:,0].clone()))
        assert logits.grad[:,0]<0 and (logits.grad[:,1:]>0).all()
    torch.testing.assert_close(values[0][0],values[1][0])
    torch.testing.assert_close(values[0][1],values[1][1])
    empty=balanced_binary_loss(logits,labels,torch.zeros_like(labels))
    assert torch.isfinite(empty) and empty==0


def test_estimated_projection_prior_is_explicit_optional_and_not_teacher_input():
    read=CADImageReadout(width=4,points=4,stride=1)
    atlas=dict(atlas_query=torch.zeros(1,16,4),atlas_keys=torch.zeros(1,4,4),atlas_xyz=torch.zeros(1,4,3),atlas_available=torch.ones(1,4,dtype=torch.bool),atlas_temperature=.1)
    projection=torch.tensor([[[1.,1.],[2.,2.],[1.,2.],[2.,1.]]])
    out=read(atlas,(4,4),estimated_uv=projection,prior_sigma=.1)
    torch.testing.assert_close(out['cad_image_uv'],projection,atol=1e-6,rtol=0)
    assert torch.equal(read(atlas,(4,4))['cad_image_uv'],read(atlas,(4,4),estimated_uv=projection)['cad_image_uv'])
