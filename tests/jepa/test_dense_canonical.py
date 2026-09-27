from types import SimpleNamespace
import torch
from lip.unified.dense_canonical import DenseCanonicalReadout,dense_canonical_loss
from lip.unified.cad_transport import transport_targets


def fixture():
    y,x=torch.meshgrid(torch.arange(224),torch.arange(224),indexing='ij')
    geometry=torch.zeros(1,9,224,224)
    geometry[:,3]=1;geometry[:,4]=(x-111.5)/224;geometry[:,5]=(y-111.5)/224
    dense=torch.randn(1,16,224,224)
    fallback=torch.randn(1,5,224,224,requires_grad=True)
    valid=torch.ones(1,256,dtype=torch.bool)
    return DenseCanonicalReadout(),dense,fallback,geometry,valid


def test_identity_preserves_dense_surface_and_leaves_depth_validity_unchanged():
    model,dense,fallback,geometry,valid=fixture()
    surface,result=model(dense,fallback,geometry,valid)
    assert torch.equal(surface[:,:3],geometry[:,4:7])
    assert torch.equal(surface[:,3:],fallback[:,3:])
    assert not result['flow'].any() and result['gate'].all()
    surface[:,0].mean().backward()
    assert model.head[-1].weight.grad[:2].norm()>0
    assert torch.isfinite(model.head[-1].weight.grad).all()


def test_missing_reference_uses_original_prediction_exactly():
    model,dense,fallback,geometry,valid=fixture()
    geometry[:,3]=0
    surface,result=model(dense,fallback,geometry,valid)
    assert torch.equal(surface,fallback) and not result['gate'].any()


def test_teacher_projection_is_identity_and_correspondence_loss_ignores_predicted_gate():
    model,dense,fallback,geometry,valid=fixture()
    base=torch.eye(4)[None];base[:,2,3]=1
    k=torch.tensor([[[224.,0,111.5],[0,224.,111.5],[0,0,1.]]])
    obs=SimpleNamespace(geometry_image=geometry,base=base,diameter=torch.ones(1),cad_valid=valid)
    supported=torch.ones(1,1,224,224,dtype=torch.bool)
    target=SimpleNamespace(cad_geometry_xyz=geometry[:,4:7],cad_geometry_valid=supported,
                           geometry_real_weight=supported,geometry_proxy_weight=~supported)
    label=transport_targets(target.cad_geometry_xyz,geometry,base,obs.diameter,k,valid)
    torch.testing.assert_close(label['flow'],torch.zeros_like(label['flow']),atol=2e-5,rtol=0)
    assert label['supported'].all()
    with torch.no_grad():
        model.head[-1].bias[0]=.01
        model.head[-1].bias[2]=-10
    _,result=model(dense,fallback,geometry,valid)
    assert not result['gate'].any()
    loss,_=dense_canonical_loss([result,result],target,obs,k,~supported)
    loss.backward()
    assert model.head[-1].bias.grad[0].abs()>0
    assert model.head[-1].bias.grad[2].abs()>0
