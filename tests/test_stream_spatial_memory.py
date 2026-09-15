from dataclasses import replace
import torch
import pytest
from lip.models.stream_spatial_memory import SpatialRKTracker,SpatialCache,dense_source_xy,select_patches
from lip.models.stream_rk import RKTracker,AnchorBank
from test_stream_rk import features,meta


def test_dense_source_coordinates_preserve_affine_pixel_centers():
    xy=torch.stack(torch.meshgrid(torch.arange(4)+.5,torch.arange(4)+.5,indexing='ij'),-1).flip(-1).reshape(1,16,2)
    transform=torch.tensor([[.2,.03],[-.02,.4]]);offset=torch.tensor([.1,.2])
    xy=xy@transform+offset;actual=dense_source_xy(xy,14)
    cell=(torch.arange(14)+.5)*4/14
    target=torch.stack(torch.meshgrid(cell,cell,indexing='ij'),-1).flip(-1).reshape(1,-1,2)@transform+offset
    torch.testing.assert_close(actual,target,rtol=0,atol=3e-7)


def test_zero_spatial_residual_preserves_trained_rk_parent():
    torch.manual_seed(7);parent=RKTracker(True,True).eval()
    for p in (parent.head[-1].weight,parent.anchor_attn.out_proj.weight,parent.readout.cross_attn.out_proj.weight):torch.nn.init.normal_(p,std=.02)
    parent.reliability_strength.data.fill_(.1);parent.update_strength.data.fill_(.1)
    model=SpatialRKTracker(dense_side=2).eval();r=model.load_state_dict(parent.state_dict(),strict=False)
    assert not r.unexpected_keys and all(n.startswith(('patch_attention.','patch_gate.','patch_pose.')) for n in r.missing_keys)
    f=features();a=b=None
    with torch.no_grad():
        for i in range(16):
            expected,a=parent(f,meta(i),a);actual,b=model(f,meta(i),b)
            assert torch.equal(expected['latent'],actual['latent'])
            assert torch.equal(expected['pose_centered'],actual['pose_centered'])
    assert b.patches.shape==(2,4,4,256) and actual['spatial_tokens_read'].min()>0
    assert actual['spatial_update_norm'].sum()==0
    with pytest.raises(ValueError):model(f,meta(16),a)


def test_patch_selection_follows_anchor_identity_and_keeps_old_values():
    f=features();bank=AnchorBank.empty(torch.zeros(2,256),4);patches=q=None
    for i in range(20):
        next_bank=bank.insert(torch.full((2,256),float(i)),torch.ones(2),meta(i),f,4,64)
        old=None if patches is None else patches.clone()
        new_patches,new_q=select_patches(bank,next_bank,patches,q,torch.full((2,9,256),float(i)),torch.ones(2,9),meta(i))
        if old is not None:assert torch.equal(old,patches)
        assert torch.equal(new_patches[:,:,0,0][next_bank.valid],next_bank.frame_id[next_bank.valid].float())
        bank,patches,q=next_bank,new_patches,new_q


def test_spatial_gradients_and_causal_mask():
    torch.manual_seed(6);model=SpatialRKTracker(dense_side=2).train();torch.nn.init.normal_(model.head[-1].weight,std=.02)
    f=features();cache=None
    for i in range(12):
        out,cache=model(f,meta(i),cache)
        if i==7:cache=cache.detach()
    out['pose_centered'].sum().backward()
    for name,p in model.named_parameters():
        if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert model.patch_attention.out_proj.weight.grad.abs().sum()>0
    future=replace(cache.anchors,frame_id=torch.full_like(cache.anchors.frame_id,100),timestamp=torch.full_like(cache.anchors.timestamp,100.))
    value,used=model.patch_read(out['latent_object'],future,cache.patches,cache.patch_quality,f,meta(12))
    assert used.sum()==0 and value.abs().sum()==0


def test_online_spatial_cache_survives_correction_and_clears_on_reset():
    from test_stream_geometry import fixture
    mesh,pose,k=fixture();model=SpatialRKTracker(dense_side=2).eval();state=model.initialize(pose,mesh,k,'x',0.,image_shape=(64,64))
    for i in range(1,11):
        p,n=model.step(torch.zeros(3,64,64),torch.zeros(1,64,64),i/30,state,image_size=32)
        assert p['status']=='ok';state=model.commit(p,n)
    before=state.cache.patches.clone();fixed=model.correct(state,pose)
    assert fixed.cache is state.cache and torch.equal(before,state.cache.patches)
    fresh=model.correct(state,pose,relocalization=True)
    assert not fresh.cache.metadata and getattr(fresh.cache,'patches',None) is None
