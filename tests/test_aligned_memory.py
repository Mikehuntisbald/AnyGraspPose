from dataclasses import replace
import torch
import pytest
from lip.models.aligned_memory import pool_memory_geometry,canonical_distance,AlignedMemoryReadout
from lip.models.stream_rk import AnchorBank
from lip.engine.stream_state import FrameMeta


def fixture(n=9,q=4):
    torch.manual_seed(41);b=2;slots=4
    objects=torch.randn(b,256);queries=torch.randn(b,q,256);keys=torch.randn(b,slots,n,256)
    qg=torch.zeros(b,q,5);kg=torch.zeros(b,slots,n,5)
    qg[:,:,:3]=torch.randn(b,q,3)*.1;kg[:,:,:,:3]=torch.randn(b,slots,n,3)*.1;qg[...,4]=1;kg[...,4]=1
    bank=AnchorBank.empty(objects,slots)
    bank=replace(bank,valid=torch.ones(b,slots,dtype=torch.bool),frame_id=torch.tensor([[0,4,8,12],[0,4,8,12]]),
        timestamp=torch.tensor([[0.,.1,.2,.3],[0.,.1,.2,.3]],dtype=torch.float64),stream_tag=torch.tensor([[3]*4,[9]*4]))
    meta=FrameMeta(torch.tensor([1.,1.],dtype=torch.float64),torch.tensor([20,20]),torch.tensor([3,9]),torch.ones(b,17,dtype=torch.bool),torch.zeros(b,17))
    return objects,queries,qg,keys,kg,bank,meta


def test_pooling_uses_only_rendered_cad_coordinates_and_preserves_variance():
    g=torch.zeros(1,9,2,2);g[:,3]=torch.tensor([[1.,1.],[0.,0.]])
    g[:,4]=torch.tensor([[.1,.3],[900.,900.]])
    actual=pool_memory_geometry(g,1)[0,0]
    torch.testing.assert_close(actual,torch.tensor([.2,0.,0.,.01,.5]))
    g[:,3]=0;assert pool_memory_geometry(g,1).abs().sum()==0


def test_canonical_distance_matches_points_after_joint_rigid_coordinate_change():
    q=torch.zeros(1,2,5);q[0,:,0]=torch.tensor([0.,.2]);k=q.clone()
    distance=canonical_distance(q,k);assert distance[0,0,0]==0 and distance[0,0,1]>0
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    q2=q.clone();k2=k.clone();q2[:,:,:3]=q[:,:,:3]@rotation.T+.3;k2[:,:,:3]=k[:,:,:3]@rotation.T+.3
    torch.testing.assert_close(canonical_distance(q2,k2),distance)
    wider=k.clone();wider[...,3]=.1;assert canonical_distance(q,wider)[0,0,1]<distance[0,0,1]


def test_zero_residual_and_valid_gradient_start():
    model=AlignedMemoryReadout();args=fixture();out,diagnostics=model(*args)
    assert out.abs().sum()==0 and diagnostics['usable_tokens'].tolist()==[36,36]
    out.sum().backward()
    assert model.attention.out_proj.weight.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad();torch.nn.init.normal_(model.attention.out_proj.weight,std=.02)
    out,_=model(*args);out.square().sum().backward()
    assert model.geometry_strength.grad.abs().sum()>0 and model.query_norm.weight.grad.abs().sum()>0


def test_future_cross_stream_and_foreground_masks_are_causal_with_finite_fallback():
    model=AlignedMemoryReadout();torch.nn.init.normal_(model.attention.out_proj.weight,std=.02);args=list(fixture())
    bank=args[5];invalid=replace(bank,frame_id=torch.full_like(bank.frame_id,100))
    args[5]=invalid;out,diag=model(*args);assert out.abs().sum()==0 and diag['usable_tokens'].sum()==0
    args[5]=replace(bank,stream_tag=bank.stream_tag+1);out,_=model(*args);assert out.abs().sum()==0
    args[5]=replace(bank,timestamp=torch.ones_like(bank.timestamp));out,_=model(*args);assert out.abs().sum()==0
    args[5]=bank;args[4][...,4]=0;out,_=model(*args);assert out.abs().sum()==0 and torch.isfinite(out).all()
    out.sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_key_permutation_with_geometry_and_batch_independence():
    model=AlignedMemoryReadout().eval();torch.nn.init.normal_(model.attention.out_proj.weight,std=.02);args=list(fixture())
    with torch.no_grad():
        reference,_=model(*args);order=torch.tensor([4,2,0,8,1,7,6,3,5]);args[3]=args[3][:,:,order];args[4]=args[4][:,:,order]
        permuted,_=model(*args);torch.testing.assert_close(permuted,reference,rtol=1e-5,atol=1e-6)
        args[3][1].mul_(10);changed,_=model(*args);torch.testing.assert_close(changed[0],permuted[0],rtol=0,atol=0)


def test_no_current_foreground_cannot_generate_a_memory_update():
    model=AlignedMemoryReadout();torch.nn.init.normal_(model.attention.out_proj.weight,std=.02);args=list(fixture());args[2][...,4]=0
    out,diag=model(*args);assert out.abs().sum()==0 and diag['usable_queries'].sum()==0 and torch.isfinite(out).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('queries',[16,196])
def test_bf16_real_cache_shape_backward_is_finite(queries):
    args=list(fixture(n=196,q=queries));args[:5]=[x.cuda() for x in args[:5]]
    bank=args[5];args[5]=AnchorBank(*(getattr(bank,n).cuda() for n in bank.__dataclass_fields__))
    meta=args[6];args[6]=FrameMeta(*(getattr(meta,n).cuda() for n in meta.__dataclass_fields__))
    model=AlignedMemoryReadout().cuda();torch.nn.init.normal_(model.attention.out_proj.weight,std=.02)
    with torch.autocast('cuda',dtype=torch.bfloat16):out,_=model(*args)
    out.float().square().mean().backward()
    assert torch.isfinite(out).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.geometry_strength.grad.abs().sum()>0 and model.query_norm.weight.grad.abs().sum()>0
