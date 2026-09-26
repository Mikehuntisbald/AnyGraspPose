from dataclasses import fields
from types import SimpleNamespace
import pytest
import torch
from lip.unified.features import TeacherTargets
from lip.unified.supervision_quality import quarantine_real_geometry


def example():
    shape=(1,1,32,32)
    depth=torch.full(shape,.8)
    real=torch.zeros(shape,dtype=torch.bool);real[:,:,8:24,8:24]=True
    proxy=torch.zeros_like(real);proxy[:,:,2:6,2:6]=True
    target=TeacherTargets(**{f.name:None for f in fields(TeacherTargets)})
    # Dataclass is frozen; use replace to build a complete mask/target fixture.
    from dataclasses import replace
    target=replace(target,surface_depth_m=depth.clone(),surface_xyz=torch.ones(1,3,32,32),
        geometry_real_weight=real,geometry_proxy_weight=proxy,geometry_weight=real|proxy,
        geometry_visible_weight=real,geometry_hidden_weight=proxy,geometry_valid_label=torch.ones(shape),
        cad_geometry_depth_m=depth.clone(),cad_geometry_valid=torch.ones_like(real))
    visible=~proxy
    return target,[SimpleNamespace(depth=depth)],visible


def test_bad_depth_is_unknown_without_cad_substitution_or_proxy_change():
    target,scenes,visible=example()
    scenes[0].depth[:,:,16,16]=.95
    target.surface_depth_m[:,:,16,16]=.95
    corrected=quarantine_real_geometry(target,scenes,visible,.03)
    assert not corrected.geometry_real_weight[:,:,14:19,14:19].any()
    assert torch.isnan(corrected.geometry_valid_label[:,:,16,16]).all()
    assert corrected.surface_depth_m is target.surface_depth_m
    assert corrected.surface_depth_m[0,0,16,16]==.95
    assert corrected.surface_xyz is target.surface_xyz
    assert torch.equal(corrected.geometry_proxy_weight,target.geometry_proxy_weight)
    assert torch.equal(corrected.geometry_hidden_weight,target.geometry_hidden_weight)
    assert torch.equal(corrected.geometry_weight,corrected.geometry_real_weight|target.geometry_proxy_weight)
    assert corrected.geometry_real_weight[0,0,10,10]
    assert target.geometry_real_weight[0,0,16,16]  # historical labels immutable


def test_invalid_depth_and_cad_boundary_cannot_train_visible_correspondence():
    target,scenes,visible=example()
    scenes[0].depth[:,:,16,16]=0
    target.cad_geometry_valid[:,:,:,0:10]=False
    corrected=quarantine_real_geometry(target,scenes,visible,.03)
    assert not corrected.real_geometry_eligible[:,:,14:19,14:19].any()
    assert not corrected.real_geometry_eligible[:,:,:,:12].any()
    assert not corrected.geometry_real_weight[:,:,:,:12].any()


@pytest.mark.parametrize('threshold',[0,-.01,.06])
def test_requires_explicit_bounded_threshold(threshold):
    target,scenes,visible=example()
    with pytest.raises(ValueError):quarantine_real_geometry(target,scenes,visible,threshold)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires actual CUDA autocast')
def test_transport_teacher_is_invariant_to_outer_bfloat16():
    from lip.unified.cad_transport import transport_targets,pixel_grid
    from lip.geometry.so3 import exp
    grid=pixel_grid(1,224,224,'cuda')
    k=torch.tensor([[[500.,0,111.5],[0,500.,111.5],[0,0,1.]]],device='cuda')
    xyz=torch.cat(((grid-111.5)/500,torch.zeros_like(grid[:,:1])),1)
    base=torch.eye(4,device='cuda')[None];base[:,2,3]=1
    geometry=torch.zeros(1,9,224,224,device='cuda');geometry[:,3]=1;geometry[:,4:7]=xyz
    valid=torch.ones(1,256,device='cuda',dtype=torch.bool);diameter=torch.ones(1,device='cuda')
    for rotation in (torch.zeros(3,device='cuda'),torch.tensor([.13,-.21,.07],device='cuda')):
        base[:,:3,:3]=exp(rotation)
        expected=transport_targets(xyz,geometry,base,diameter,k,valid)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            actual=transport_targets(xyz,geometry,base,diameter,k,valid)
        for name in expected:assert torch.equal(actual[name],expected[name]),name
        if not rotation.any():assert expected['flow'].abs().max()<1e-4
