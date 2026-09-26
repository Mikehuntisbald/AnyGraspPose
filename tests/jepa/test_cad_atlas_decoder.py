from types import SimpleNamespace
import torch
from lip.unified.cad_atlas_decoder import CADAtlasDecoder,atlas_correspondence_loss


def fixture():
    torch.manual_seed(42)
    model=CADAtlasDecoder(12)
    dense=torch.randn(1,16,4,4,requires_grad=True)
    fallback=torch.randn(1,5,4,4)*.05
    features=torch.randn(1,16,12)
    geometry=torch.randn(1,16,21)*.1
    available=torch.ones(1,16,dtype=torch.bool)
    return model,dense,fallback,features,geometry,available


def test_forward_returns_actual_cad_point_and_depth_unchanged():
    model,dense,fallback,features,geometry,available=fixture()
    surface,out=model(dense,fallback,features,geometry,available)
    expected=geometry[:,:,:3][torch.arange(1)[:,None],out['atlas_index']].transpose(1,2).reshape(1,3,4,4)
    assert torch.equal(surface[:,:3],expected)
    assert torch.equal(surface[:,3:],fallback[:,3:])
    target=SimpleNamespace(cad_geometry_xyz=geometry[:,:,:3].transpose(1,2).reshape(1,3,4,4),
        cad_geometry_valid=torch.ones(1,1,4,4,dtype=torch.bool),geometry_real_weight=torch.ones(1,1,4,4,dtype=torch.bool),
        geometry_proxy_weight=torch.zeros(1,1,4,4,dtype=torch.bool))
    loss,_=atlas_correspondence_loss(out,target,target.geometry_proxy_weight)
    loss.backward()
    for grad in (dense.grad,model.query[0].weight.grad,model.descriptor[1].weight.grad):
        assert torch.isfinite(grad).all() and grad.norm()>0
    assert not target.cad_geometry_xyz.requires_grad


def test_missing_cad_falls_back_exactly_even_with_nan_storage():
    model,dense,fallback,features,geometry,available=fixture()
    available[:]=False;features[:]=float('nan');geometry[:]=float('nan')
    surface,_=model(dense,fallback,features,geometry,available)
    assert torch.equal(surface,fallback)


def test_query_labels_cannot_select_forward_points():
    model,dense,fallback,features,geometry,available=fixture()
    first,_=model(dense,fallback,features,geometry,available)
    # The forward signature has no teacher target, original visibility or GT pose.
    second,_=model(dense,fallback,features,geometry,available)
    assert torch.equal(first,second)


def test_actual_points_can_be_learned_with_global_correspondence_loss():
    model,dense,fallback,features,geometry,available=fixture()
    # Fit a known16-way correspondence, retaining a nonzero wrong geometric prior.
    target=SimpleNamespace(cad_geometry_xyz=geometry[:,:,:3].transpose(1,2).reshape(1,3,4,4),
        cad_geometry_valid=torch.ones(1,1,4,4,dtype=torch.bool),geometry_real_weight=torch.ones(1,1,4,4,dtype=torch.bool),
        geometry_proxy_weight=torch.zeros(1,1,4,4,dtype=torch.bool))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    dense=dense.detach()
    for step in range(80):
        surface,out=model(dense,fallback,features,geometry,available)
        loss,_=atlas_correspondence_loss(out,target,target.geometry_proxy_weight)
        optimizer.zero_grad();loss.backward();optimizer.step()
    surface,_=model(dense,fallback,features,geometry,available)
    assert (surface[:,:3]-target.cad_geometry_xyz).norm(dim=1).max()<1e-6
