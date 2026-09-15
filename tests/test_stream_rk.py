import pytest
import torch
from dataclasses import replace
from lip.models.stream_rk import RKTracker, AnchorBank, observation_support
from lip.models.stream_tracker import StreamTracker
from lip.engine.stream_state import FrameMeta


def meta(i,b=2,tag=0):
    return FrameMeta(torch.full((b,),i/30,dtype=torch.float64),torch.full((b,),i,dtype=torch.long),
        torch.arange(b,dtype=torch.long)+tag,torch.ones(b,17,dtype=torch.bool),torch.zeros(b,17))

def features(b=2):
    pose=torch.eye(4).repeat(b,1,1);pose[:,2,3]=.6
    g=torch.zeros(b,9,32,32);g[:,1]=1;g[:,3]=1;g[:,8]=1
    return dict(rgb=torch.randn(b,3,32,32),geometry=g,state_input=torch.randn(b,24),
        source_xy=torch.randn(b,16,2),T_base_centered=pose,object_diameter_m=torch.full((b,),.1),mesh_center=torch.zeros(b,3))


def test_support_unknown_is_neutral_and_occlusion_is_soft():
    f=features();g=f['geometry']
    tokens,good=observation_support(g);assert torch.all(good==1)
    g[:,7]=-.2
    _,bad=observation_support(g);assert torch.all((bad>.25)&(bad<.5))
    g[:,1]=0;g[:,8]=0
    tokens,unknown=observation_support(g);assert torch.all(unknown==.5) and torch.all(tokens==.5)
    g[:,3]=0
    assert torch.all(observation_support(g)[1]==.5)


def test_anchor_quality_spacing_source_identity_and_immutability():
    f=features();bank=AnchorBank.empty(torch.zeros(2,256),4)
    for i in range(20):
        old=bank;before=old.pose.clone()
        quality=torch.tensor([1. if i==0 else .3, 1. if i==4 else .3])
        bank=bank.insert(torch.full((2,256),float(i)),quality,meta(i),f,4,64)
        assert torch.equal(old.pose,before)
    assert 0 in bank.frame_id[0,bank.valid[0]] and 4 in bank.frame_id[1,bank.valid[1]]
    for lane in range(2):
        ids=bank.frame_id[lane,bank.valid[lane]]
        assert all(abs(int(a)-int(b))>=4 for j,a in enumerate(ids) for b in ids[j+1:])
    changed=bank.insert(torch.zeros(2,256),torch.ones(2),meta(20,tag=100),f,4,64)
    assert changed.valid.sum(1).tolist()==[1,1]
    assert torch.equal(changed.stream_tag[:,0],torch.tensor([100,101]))
    expired=bank.insert(torch.zeros(2,256),torch.ones(2),meta(100),f,4,64)
    assert expired.valid.sum(1).tolist()==[1,1]


@pytest.mark.parametrize('r,k',[(False,False),(True,False),(False,True),(True,True)])
def test_all_arms_start_at_exact_parent_function(r,k):
    torch.manual_seed(7)
    parent=StreamTracker('stream_dual_cross_residual',dropout=0.).eval()
    # Nonzero existing branch simulates an already trained parent.
    torch.nn.init.normal_(parent.readout.cross_attn.out_proj.weight,std=.01)
    torch.nn.init.normal_(parent.head[-1].weight,std=.01)
    model=RKTracker(r,k).eval()
    result=model.load_state_dict(parent.state_dict(),strict=False)
    assert not result.unexpected_keys
    # Equal requires_grad flags select the same CPU attention projection
    # kernels; otherwise PyTorch differs by approximately 1e-6 in the encoder.
    for name,p in parent.named_parameters():p.requires_grad_(dict(model.named_parameters())[name].requires_grad)
    f=features();a=b=None
    with torch.no_grad():
        for i in range(14):
            expected,a=parent(f,meta(i),a)
            actual,b=model(f,meta(i),b)
            assert torch.equal(expected['pose_centered'],actual['pose_centered'])
            assert torch.equal(expected['latent'],actual['latent'])
    if k:
        assert actual['anchors_read'].min()>0 and b.anchors.valid.any()
    with pytest.raises(ValueError):model(f,meta(15),replace(b,variant='wrong'))


def test_active_modules_receive_gradients_and_anchor_selection_is_causal():
    torch.manual_seed(4);model=RKTracker(True,True).train()
    torch.nn.init.normal_(model.head[-1].weight,std=.01)
    f=features();cache=None
    for i in range(12):
        out,cache=model(f,meta(i),cache)
        if i==7:cache=cache.detach()
    out['pose_centered'].sum().backward()
    for n,p in model.named_parameters():
        if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),n
    assert model.anchor_attn.out_proj.weight.grad.abs().sum()>0
    assert model.reliability_strength.grad is not None
    future=replace(cache.anchors,frame_id=torch.full_like(cache.anchors.frame_id,100),
                   timestamp=torch.full_like(cache.anchors.timestamp,100.))
    h,used=model.anchor_read(out['latent_object'][:,None],future,meta(12))
    assert used.sum()==0 and h.abs().sum()==0
    assert cache.detach().anchors.latent.grad_fn is None


def test_equal_quality_keeps_spaced_anchors_instead_of_only_latest_frame():
    f=features();bank=AnchorBank.empty(torch.zeros(2,256),4)
    for i in range(24):bank=bank.insert(torch.full((2,256),float(i)),torch.ones(2),meta(i),f,4,64)
    assert bank.valid.sum(1).tolist()==[4,4]
    assert sorted(bank.frame_id[0].tolist())==[8,12,16,20]


def test_online_correction_preserves_anchors_and_relocalization_clears_them():
    from test_stream_geometry import fixture
    mesh,pose,k=fixture();model=RKTracker(True,True).eval()
    state=model.initialize(pose,mesh,k,'test-stream',0.,image_shape=(64,64))
    for i in range(1,14):
        proposal,candidate=model.step(torch.zeros(3,64,64),torch.zeros(1,64,64),i/30,state,image_size=32)
        assert proposal['status']=='ok';state=model.commit(proposal,candidate)
    bank=state.cache.anchors;stored=bank.pose.clone();latent=bank.latent.clone()
    corrected=model.correct(state,pose)
    assert corrected.cache is state.cache and torch.equal(bank.pose,stored) and torch.equal(bank.latent,latent)
    reset=model.correct(state,pose,relocalization=True)
    assert getattr(reset.cache,'anchors',None) is None and not reset.cache.metadata
    assert reset.generation==state.generation+1
