import torch
from lip.unified.cad_atlas_decoder import CADAtlasDecoder


def fixture():
    torch.manual_seed(71)
    model=CADAtlasDecoder(feature_dim=4)
    dense=torch.randn(1,16,2,2)
    fallback=torch.zeros(1,5,2,2,requires_grad=True)
    features=torch.randn(1,16,4)
    geometry=torch.zeros(1,16,21)
    geometry[0,:8,0]=torch.linspace(.001,.008,8)
    geometry[0,8:,0]=torch.linspace(.3,.4,8)
    return model,dense,fallback,features,geometry,torch.ones(1,16,dtype=torch.bool)


def test_appearance_cannot_choose_distant_surface_and_xyz_trains_prior():
    model,dense,fallback,features,geometry,available=fixture()
    # Force descriptor evidence to favor the distant surface. Verify the old
    # unrestricted read actually fails this case before testing the constraint.
    model.query=torch.nn.Conv2d(16,32,1)
    model.descriptor=torch.nn.Linear(4,32,bias=False)
    with torch.no_grad():
        model.query.weight.zero_();model.query.bias.zero_();model.query.bias[0]=1
        model.descriptor.weight.zero_();model.descriptor.weight[0,0]=1
        for parameter in model.geometry.parameters():parameter.zero_()
    features.zero_();features[0,:8,0]=-1;features[0,8:,0]=1
    unconstrained,_=model(dense,fallback,features,geometry,available)
    assert (unconstrained[:,0]>=.3).all()
    model.local_surface_projection=True
    surface,output=model(dense,fallback,features,geometry,available)
    assert (surface[:,0]<=.008).all() and (surface[:,0]>=.001).all()
    assert (output['atlas_index']<8).all()
    surface[:,:3].sum().backward()
    torch.testing.assert_close(fallback.grad[:,:3],torch.ones_like(fallback.grad[:,:3]))
    assert torch.equal(surface[:,3:],fallback[:,3:])


def test_no_available_cad_preserves_fallback_and_default_is_unchanged():
    model,dense,fallback,features,geometry,available=fixture()
    assert model.local_surface_projection is False
    model.local_surface_projection=True
    surface,_=model(dense,fallback,features,geometry,torch.zeros_like(available))
    assert torch.equal(surface,fallback)
