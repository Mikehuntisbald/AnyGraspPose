from types import SimpleNamespace
import pytest
import torch
from lip.unified.full_recovery_metrics import recovery_metrics


def fixture():
    h=16;y,x=torch.meshgrid(torch.arange(h),torch.arange(h),indexing='ij')
    rays=torch.stack(((x-7.5)/100,(y-7.5)/100,torch.ones(h,h)),0)[None]
    d=torch.tensor([.2]);z=torch.ones(1,1,h,h)
    xyz=(rays-torch.tensor([0.,0.,1.])[None,:,None,None])/.2
    t=SimpleNamespace(camera_rays=rays,camera_rotation=torch.eye(3)[None],camera_translation_d=torch.tensor([[0.,0.,5.]]),
        cad_geometry_xyz=xyz,surface_depth_m=z,geometry_valid_label=z)
    o=dict(surface_xyz=xyz.clone(),surface_depth_m=z.clone(),geometry_valid_logits=z*10)
    k=torch.tensor([[[100.,0.,7.5],[0.,100.,7.5],[0.,0.,1.]]]);mask=torch.ones_like(z,dtype=torch.bool)
    return o,t,d,k,mask


def test_exact_plane_has_zero_physical_and_correspondence_error():
    o,t,d,k,m=fixture();r=recovery_metrics(o,t,d,k,{'real':m})[0]['real']
    assert r['canonical_xyz_mm']==0 and r['depth_mm']==0 and r['camera_xyz_mm']==0
    assert r['identity_reprojection_px']<1e-5 and r['camera_normal_deg']<.01
    assert r['joint_xyz10mm_depth5mm']==1 and r['pixels']==256


def test_sensor_target_ownership_and_depth_units_are_preserved():
    o,t,d,k,m=fixture();sensor=t.surface_depth_m+.02;o['surface_depth_m']=sensor.clone()
    r=recovery_metrics(o,t,d,k,{'proxy':m,'visible':m},visible_depth_m=sensor)[0]
    assert r['visible']['depth_mm']==0
    assert abs(r['proxy']['depth_mm']-20)<1e-3
    assert r['proxy']['camera_xyz_mm']>=r['proxy']['depth_mm']
    assert r['proxy']['canonical_xyz_mm']==0  # canonical identity is a different quantity
    with pytest.raises(ValueError):recovery_metrics(o,t,d,k,{'visible':m})


def test_confidence_cannot_hide_bad_geometry_and_empty_regions_remain_none():
    o,t,d,k,m=fixture();o['surface_depth_m']+=.03
    high=recovery_metrics(o,t,d,k,{'real':m,'proxy':~m})[0]
    o['geometry_valid_logits'].fill_(-10)
    low=recovery_metrics(o,t,d,k,{'real':m,'proxy':~m})[0]
    assert low['real']['depth_mm']==high['real']['depth_mm']
    assert low['real']['valid_recall']==0 and low['real']['joint_xyz10mm_depth5mm']==0
    assert low['proxy'] is None


def test_nonfinite_predictions_fail_instead_of_being_omitted():
    o,t,d,k,m=fixture();o['surface_xyz'][0,0,0,0]=float('nan')
    with pytest.raises(ValueError):recovery_metrics(o,t,d,k,{'real':m})
