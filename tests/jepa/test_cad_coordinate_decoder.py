import torch
from lip.unified.cad_coordinate_decoder import CADCoordinateDecoder,dense_anchor_loss


def test_dense_cad_reader_gradients_and_absent_reference_fallback():
    torch.manual_seed(42)
    m=CADCoordinateDecoder(12)
    levels=[torch.randn(1,256,256) for _ in range(4)]
    valid=torch.ones(1,256,dtype=torch.bool)
    features=torch.randn(1,8,12);geometry=torch.randn(1,8,21)*.1
    available=torch.ones(1,8,dtype=torch.bool)
    out,logits=m(levels,valid,features,geometry,available)
    target=torch.randn(1,3,224,224)*.1;mask=torch.ones(1,1,224,224,dtype=torch.bool)
    loss=(out[:,:3]-target).square().mean()+dense_anchor_loss(logits,target,mask,geometry[:,:,:3],available)
    loss.backward()
    for module in (m.query,m.descriptor,m.geometry,m.offset,m.dpt.projects[0]):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
    with torch.no_grad():
        expected=m.dpt(levels,valid)
        result,_=m(levels,valid,features*float('nan'),geometry*float('nan'),~available)
    torch.testing.assert_close(result,expected)


def test_anchor_targets_preserve_canonical_identity_and_ignore_empty_masks():
    xyz=torch.zeros(1,3,224,224);xyz[:,0]=.2
    anchors=torch.tensor([[[-.2,0,0],[.2,0,0]]]);mask=torch.ones(1,1,224,224,dtype=torch.bool)
    available=torch.ones(1,2,dtype=torch.bool)
    correct=torch.tensor([-10.,10.])[None,None].repeat(1,3136,1)
    assert dense_anchor_loss(correct,xyz,mask,anchors,available)<1e-5
    assert dense_anchor_loss(-correct,xyz,mask,anchors,available)>10
    assert dense_anchor_loss(correct,xyz,~mask,anchors,available)==0
