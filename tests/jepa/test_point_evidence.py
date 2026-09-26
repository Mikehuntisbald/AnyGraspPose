from types import SimpleNamespace
import torch
from lip.unified.point_evidence import PointEvidenceHead,point_evidence_features,point_evidence_targets,BASE_DIM,FEATURE_DIM
from lip.unified.flow_reconstruction import FlowReconstruction,template_points


def fixture():
    torch.manual_seed(58)
    geometry=torch.zeros(1,9,224,224);geometry[:,1]=geometry[:,3]=1
    y,x=torch.meshgrid(torch.arange(224),torch.arange(224),indexing='ij')
    geometry[:,4]=(x-111.5)/224;geometry[:,5]=(y-111.5)/224
    valid=torch.ones(1,256,dtype=torch.bool);patch=torch.randn(1,256,256)
    return geometry,valid,patch,torch.randn_like(patch),torch.randn_like(patch)


def test_control_is_invariant_to_observed_features_candidate_can_read_them():
    x=torch.randn(4,FEATURE_DIM,requires_grad=True);changed=x.detach().clone();changed[:,BASE_DIM:]+=3
    control=PointEvidenceHead(False);observed=PointEvidenceHead(True);observed.load_state_dict(control.state_dict())
    assert torch.equal(control(x),control(changed))
    assert not torch.equal(observed(x),observed(changed))
    observed(x).square().mean().backward();assert x.grad[:,BASE_DIM:].norm()>0


def test_evidence_cannot_change_flow_or_reconstruction_write():
    geometry,valid,patch,real,cad=fixture();m=FlowReconstruction();reference=template_points(geometry,valid)
    before,old=m(patch,real,cad,geometry,valid,reference)
    m.point_evidence_head=PointEvidenceHead(True)
    after,new=m(patch,real,cad,geometry,valid,reference)
    assert torch.equal(before,after) and torch.equal(old['uv'],new['uv'])
    assert torch.equal(old['support_logits'],new['support_logits'])
    assert new['point_evidence_features'].shape==(1,256,FEATURE_DIM)
    assert not new['point_evidence_features'].requires_grad
    assert torch.equal(old['visible_logits'],new['legacy_visible_logits'])


def test_visible_point_and_usable_predicted_measurement_are_distinct_labels():
    geometry,valid,_,_,_=fixture();reference=template_points(geometry,valid)
    mask=torch.ones(1,1,224,224,dtype=torch.bool)
    target=SimpleNamespace(camera_rotation=torch.eye(3)[None],camera_translation_d=torch.tensor([[0.,0.,1.]]),
        cad_geometry_valid=mask,cad_geometry_xyz=geometry[:,4:7],cad_geometry_depth_m=torch.full((1,1,224,224),.5),
        geometry_real_weight=~mask,geometry_proxy_weight=~mask,real_geometry_eligible=None)
    obs=SimpleNamespace(diameter=torch.tensor([.5]),measured_depth_m=torch.full((1,1,224,224),.5))
    k=torch.tensor([[[224.,0,111.5],[0,224.,111.5],[0,0,1.]]])
    output=dict(flow_reference=reference,flow_rounds=[dict(uv=reference['uv'].clone())])
    good,_=point_evidence_targets(output,target,obs,mask,k)
    assert (good==1).all()
    output['flow_rounds'][0]['uv']+=8
    bad,_=point_evidence_targets(output,target,obs,mask,k)
    assert (bad[...,0]==1).all() and (bad[...,1]==0).all()
    output['flow_rounds'][0]['uv']=reference['uv'].clone();obs.measured_depth_m[:]=.2
    occluder,_=point_evidence_targets(output,target,obs,mask,k)
    assert (occluder[...,0]==1).all() and (occluder[...,1]==0).all()
