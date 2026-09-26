from types import SimpleNamespace
import torch
from lip.unified.cad_transport import CADTransport,pixel_grid
from lip.unified.geometry_transport_objective import objective


def test_cad_identity_preserves_real_depth_supervision():
    grid=pixel_grid(1,224,224,'cpu')
    rays=torch.cat(((grid-111.5)/224.,torch.ones(1,1,224,224)),1)
    translation=torch.tensor([[0.,0.,1.]])
    xyz=rays-translation[:,:,None,None]
    raw_xyz=rays*1.02-translation[:,:,None,None]
    mask=torch.zeros(1,1,224,224,dtype=torch.bool);mask[:,:,50:170,50:170]=True
    label=torch.ones(1,1,224,224)
    target=SimpleNamespace(surface_xyz=raw_xyz,surface_depth_residual=label*.02,
        geometry_real_weight=mask,geometry_proxy_weight=mask&False,geometry_weight=mask,
        geometry_valid_label=label,visible_label=torch.zeros(1,256),support_label=torch.ones(1,256),
        camera_rotation=torch.eye(3)[None],camera_translation_d=translation,camera_rays=rays,base_depth_d=torch.ones(1),
        cad_geometry_xyz=xyz,cad_geometry_depth_m=label,cad_geometry_valid=mask)
    base=torch.eye(4)[None];base[:,2,3]=1.
    obs=SimpleNamespace(measured_depth_m=torch.zeros_like(label),diameter=torch.ones(1),base=base)
    output=dict(surface_xyz=xyz.clone().requires_grad_(),surface_depth_residual=(label*.02).requires_grad_(),
                geometry_valid_logits=label*10,evidence_logits=torch.full((1,256),-10.),support_logits=torch.full((1,256),10.))
    k=torch.tensor([[[224.,0.,111.5],[0.,224.,111.5],[0.,0.,1.]]])
    clean,_=objective(output,target,obs,mask&False,k,False,canonical_surface=True)
    old,_=objective(output,target,obs,mask&False,k,False)
    assert clean<old
    assert torch.equal(target.surface_xyz,raw_xyz) and torch.equal(target.surface_depth_residual,label*.02)
    target.surface_xyz=raw_xyz+1.
    unaffected,_=objective(output,target,obs,mask&False,k,False,canonical_surface=True)
    torch.testing.assert_close(clean,unaffected)
    target.surface_depth_residual=label*.05
    changed,_=objective(output,target,obs,mask&False,k,False,canonical_surface=True)
    assert changed>clean


def test_locked_surface_cannot_use_xyz_offset_or_low_gate_to_skip_lookup():
    head=CADTransport(surface_locked=True,mandatory_lookup=True)
    with torch.no_grad():head.head[-1].bias[2:5]=10.;head.head[-1].bias[6]=-100.
    geometry=torch.zeros(1,9,224,224);geometry[:,3]=1.
    geometry[:,4:7]=torch.rand(1,3,224,224)
    fallback=torch.randn(1,5,224,224)
    surface,info=head(torch.zeros(1,16,224,224),fallback,geometry,torch.ones(1,256,dtype=torch.bool))
    torch.testing.assert_close(surface[:,:3],geometry[:,4:7])
    assert info['transport_gate'].min()==1 and info['transport_confidence'].max()<1e-6
    dropped,_=head(torch.zeros(1,16,224,224),fallback,geometry,torch.zeros(1,256,dtype=torch.bool))
    assert torch.equal(dropped,fallback)
