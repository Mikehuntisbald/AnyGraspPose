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
