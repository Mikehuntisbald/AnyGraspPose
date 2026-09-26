import torch
from lip.unified.local_flow import LocalFlowHead,FEATURE_DIM,BASE_DIM
from lip.unified.flow_reconstruction import FlowReconstruction

def test_migration_exact_and_control_excludes_extra_evidence():
    torch.manual_seed(42)
    endpoint=FlowReconstruction().endpoint
    with torch.no_grad():endpoint[-1].weight.normal_(0,.01)
    head=LocalFlowHead();head.initialize(endpoint)
    x=torch.randn(17,FEATURE_DIM)
    torch.testing.assert_close(head(x),14*endpoint(x[:,:BASE_DIM])[:,:2].tanh())
    control=LocalFlowHead(False);control.load_state_dict(head.state_dict())
    with torch.no_grad():
        head.net[0].weight[:,BASE_DIM:].normal_(0,.1)
        control.load_state_dict(head.state_dict())
    changed=x.clone();changed[:,BASE_DIM:]+=3
    assert torch.equal(control(x),control(changed))
    assert not torch.equal(head(x),head(changed))
    assert head(x).abs().max()<=14

def test_extra_features_and_full_path_initialization():
    from test_flow_reconstruction import fixture
    from lip.unified.flow_reconstruction import template_points
    model,patch,observed,cad,geometry,valid=fixture()
    ref=template_points(geometry,valid)
    before,out=model(patch,observed,cad,geometry,valid,ref)
    model.capture_local_flow=True
    captured,result=model(patch,observed,cad,geometry,valid,ref)
    assert torch.equal(before,captured)
    assert result['local_flow_features'].shape==(1,256,FEATURE_DIM)
    model.local_flow_head=LocalFlowHead();model.local_flow_head.initialize(model.endpoint)
    after,result=model(patch,observed,cad,geometry,valid,ref)
    torch.testing.assert_close(out['uv'],result['uv'])
    assert torch.equal(out['visible_logits'],result['visible_logits'])
    assert torch.equal(out['support_logits'],result['support_logits'])


def test_deterministic_bilinear_matches_value_and_gradient():
    from lip.unified.local_flow import bilinear_descriptors
    from lip.unified.cad_image_correspondence import sample_points
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(61)
    image=torch.randn(2,64,16,16,device=device,requires_grad=True)
    xy=(torch.rand(2,47,2,device=device)*18-1).requires_grad_()
    # Reference CPU backward avoids nondeterministic CUDA grid_sample.
    reference=image.detach().cpu().requires_grad_();points=xy.detach().cpu().requires_grad_()
    expected=sample_points(reference,points)
    weight=torch.randn_like(expected)
    grad_ref=torch.autograd.grad((expected*weight).sum(),(reference,points))
    old=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        actual=bilinear_descriptors(image.flatten(2).transpose(1,2),xy)
        gradients=torch.autograd.grad((actual*weight.to(device)).sum(),(image,xy))
    finally:torch.use_deterministic_algorithms(old)
    torch.testing.assert_close(actual.cpu(),expected,atol=4e-6,rtol=2e-5)
    for got,want in zip(gradients,grad_ref):torch.testing.assert_close(got.cpu(),want,atol=3e-5,rtol=1e-4)
