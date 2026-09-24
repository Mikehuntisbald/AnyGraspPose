from types import SimpleNamespace
import torch
from lip.unified.surface_normals import normal_terms,surface_normal_loss,normal_diagnostics


def plane():
    y,x=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
    return torch.stack((x*.005,y*.005,x*0.+.2)).float()[None]


def test_correct_plane_and_reversed_orientation():
    target=plane();mask=torch.ones(1,1,16,16,dtype=torch.bool)
    for stride in (1,2):
        error,valid,_=normal_terms(target,target,mask,stride)
        assert valid.all() and error.abs().max()<1e-6
        reversed=target.clone();reversed[:,0]*=-1
        error,valid,_=normal_terms(reversed,target,mask,stride)
        assert torch.allclose(error[valid],torch.full_like(error[valid],2.))


def test_no_gradient_outside_supervised_source_and_no_teacher_gradient():
    target=plane().requires_grad_();prediction=target.detach().clone()
    prediction[:,2]+=.002*torch.arange(16)[None,:,None]
    prediction.requires_grad_();mask=torch.zeros(1,1,16,16,dtype=torch.bool);mask[:,:,3:13,3:13]=True
    teacher=SimpleNamespace(surface_xyz=target,geometry_real_weight=mask,geometry_proxy_weight=mask&False)
    loss,metrics=surface_normal_loss({'surface_xyz':prediction},teacher,{'cad_feature':.5})
    loss.backward()
    assert target.grad is None and prediction.grad.abs().sum()>0
    assert not prediction.grad[~mask.expand_as(prediction)].any()
    assert metrics['surface_normal_proxy']==0
    assert metrics['normal_angle_deg_real']>0


def test_boundary_depth_jumps_invalids_and_degenerate_targets_are_excluded():
    target=plane();target[:,:,:,8:]+=1
    mask=torch.ones(1,1,16,16,dtype=torch.bool)
    _,valid,_=normal_terms(plane(),target,mask,1)
    assert not valid[:,:,:,6:8].any() and valid[:,:,:,:5].all()
    seam=mask.clone();seam[:,:,:,8]=False
    _,wide,_=normal_terms(plane(),plane(),seam,2)
    assert not wide[:,:,:,4:9].any()  # A two-pixel stencil cannot jump an invalid pixel.
    for geometry,eligible in [(torch.zeros_like(target),mask),(target*float('nan'),mask),(target,mask&False)]:
        teacher=SimpleNamespace(surface_xyz=geometry,geometry_real_weight=eligible,geometry_proxy_weight=eligible)
        prediction=plane().requires_grad_()
        loss,_=surface_normal_loss({'surface_xyz':prediction},teacher,{'cad_feature':.5})
        assert loss==0 and loss.isfinite()
        loss.backward();assert not prediction.grad.any()
        assert normal_diagnostics({'surface_xyz':prediction},teacher,eligible) is None


def test_collapsed_prediction_still_gets_penalty():
    target=plane();prediction=torch.zeros_like(target)
    mask=torch.ones(1,1,16,16,dtype=torch.bool)
    error,valid,_=normal_terms(prediction,target,mask,1)
    assert valid.all() and torch.equal(error,torch.ones_like(error))
