import numpy as np
from lip.unified.flow_surface_audit import triangulated_transport,fit_rigid_surface,fit_metric_surface,fuse_depth_evidence


def plane():
    y,x=np.mgrid[:16,:16];uv=np.stack((x,y),-1).reshape(-1,2)*14+6.5
    xyz=np.c_[(uv-112)/224,np.zeros(256)]
    return xyz,uv,np.ones(256,dtype=bool),np.ones(256)


def test_triangle_preserves_affine_surface_and_leaves_missing_explicit():
    xyz,uv,valid,z=plane();image,mask,_=triangulated_transport(xyz,uv,valid,z)
    y,x=np.mgrid[:224,:224];truth=np.stack(((x-112)/224,(y-112)/224,np.zeros_like(x)),axis=-1)
    np.testing.assert_allclose(image[mask],truth[mask],atol=1e-7)
    assert mask[112,112] and not mask[0,0]
    valid[:]=False;_,empty,_=triangulated_transport(xyz,uv,valid,z)
    assert not empty.any()


def test_folded_flow_and_separate_sheets_are_rejected():
    xyz,uv,valid,z=plane();uv[:,0]=224-uv[:,0]
    _,mask,receipt=triangulated_transport(xyz,uv,valid,z)
    assert not mask.any() and receipt['rejected_folds']==450
    xyz,uv,valid,z=plane();xyz.reshape(16,16,3)[:,8:,2]+=1
    _,mask,receipt=triangulated_transport(xyz,uv,valid,z)
    assert receipt['rejected_discontinuities']>0 and not mask[112,112]


def test_prediction_only_rigid_fit_recovers_geometry_and_empty_falls_back():
    xyz,_,valid,_=plane();xyz*=.2
    k=np.array([[450.,0,112],[0,450,112],[0,0,1]])
    base=np.eye(4);base[2,3]=.6
    camera=xyz+np.array([.01,-.02,.5]);pixels=camera@k.T;uv=pixels[:,:2]/pixels[:,2:]
    pose,receipt=fit_rigid_surface(xyz,uv,valid.astype(float),k,base,42)
    assert receipt['accepted']
    np.testing.assert_allclose(xyz@pose[:3,:3].T+pose[:3,3],camera,atol=1e-5)
    pose,receipt=fit_rigid_surface(xyz,uv,np.zeros(256),k,base,42)
    assert not receipt['accepted'];np.testing.assert_array_equal(pose,base)


def test_metric_depth_anchor_removes_ray_scale_ambiguity():
    xyz,_,valid,_=plane();xyz*=.2
    camera=xyz+np.array([.01,-.02,.5]);k=np.array([[450.,0,112],[0,450,112],[0,0,1]])
    pixel=camera@k.T;uv=pixel[:,:2]/pixel[:,2:]
    base=np.eye(4);base[2,3]=.6
    pose,receipt=fit_metric_surface(xyz,uv,camera[:,2],valid.astype(float),k,base)
    assert receipt['accepted']
    np.testing.assert_allclose(xyz@pose[:3,:3].T+pose[:3,3],camera,atol=1e-5)


def test_soft_measured_support_does_not_disappear_at_point_five_threshold():
    recovered=np.full(3,.5);measured=np.array([.505,.2,0.])
    depth,mass,fraction=fuse_depth_evidence(recovered,measured,np.full(3,.49),np.ones(3,dtype=bool))
    assert .5<depth[0]<.505 and fraction[0]>.5
    assert fraction[1]<1e-6 and fraction[2]==0
    assert depth[2]==recovered[2] and mass[2]==.25
