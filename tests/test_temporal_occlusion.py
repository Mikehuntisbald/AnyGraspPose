import torch
from lip.data.temporal_occlusion import make_plan,composite,OccluderPlan


def test_depth_ordering_and_temporal_boundaries_do_not_modify_inputs():
    rgb=torch.zeros(3,32,32);depth=torch.ones(1,32,32);depth[:,:,:16]=.2
    before_rgb=rgb.clone();before_depth=depth.clone()
    plan=OccluderPlan(torch.ones(3,8,8), (16.,16.),(24.,24.),(0.,0.),.5,2,5,False)
    for t in [0,1,5,6]:
        r,d,mask=composite(rgb,depth,plan,t);assert r is rgb and d is depth and not mask.any()
    r,d,mask=composite(rgb,depth,plan,2)
    assert not mask[:,:,:16].any() and mask[:,:,16:].any()
    assert torch.all(d[mask]==.5) and torch.all(r[:,mask[0]]==1)
    assert torch.equal(rgb,before_rgb) and torch.equal(depth,before_depth)


def test_plan_uses_only_initial_observation_and_prior_and_is_reproducible():
    pose=torch.eye(4);pose[2,3]=.6;k=torch.tensor([[100.,0.,32.],[0.,100.,32.],[0.,0.,1.]])
    vertices=torch.tensor([[-.05,-.05,-.05],[.05,.05,.05]])
    rgb=torch.rand(3,64,64)
    a=make_plan(rgb,pose,k,vertices,.15,42,56,8,1.)
    b=make_plan(rgb,pose,k,vertices,.15,42,56,8,1.)
    assert a.center==b.center and a.size==b.size and a.start==b.start and a.end==b.end
    assert torch.equal(a.texture,b.texture) and a.start>=10 and a.end<=52
    assert make_plan(rgb,pose,k,vertices,.15,42,56,8,0.) is None


def test_invalid_depth_is_replaced_only_inside_occluder():
    rgb=torch.zeros(3,32,32);depth=torch.zeros(1,32,32)
    plan=OccluderPlan(torch.ones(3,4,4),(16.,16.),(16.,16.),(1.,0.),.4,0,4,True)
    _,d,mask=composite(rgb,depth,plan,0)
    assert mask.any() and torch.all(d[mask]==.4) and torch.all(d[~mask]==0)


def test_startup_occlusion_moves_timing_only_and_default_is_exact():
    pose=torch.eye(4);pose[2,3]=.6;k=torch.tensor([[100.,0.,32.],[0.,100.,32.],[0.,0.,1.]])
    vertices=torch.tensor([[-.05,-.05,-.05],[.05,.05,.05]]);rgb=torch.rand(3,64,64)
    for burn,total in [(0,48),(8,56)]:
        regular=make_plan(rgb,pose,k,vertices,.15,42,total,burn,1.)
        zero=make_plan(rgb,pose,k,vertices,.15,42,total,burn,1.,0.)
        startup=make_plan(rgb,pose,k,vertices,.15,42,total,burn,1.,1.)
        for field in regular.__dataclass_fields__:
            a=getattr(regular,field);b=getattr(zero,field)
            assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b
            if field not in ('start','end'):
                value=getattr(startup,field)
                assert torch.equal(a,value) if isinstance(a,torch.Tensor) else a==value
        assert startup.start==0 and startup.end==regular.end-regular.start and startup.end<=total-4
        before=rgb.clone();depth=torch.ones(1,64,64);original_depth=depth.clone()
        _,_,mask=composite(rgb,depth,startup,0)
        assert mask.any() and torch.equal(rgb,before) and torch.equal(depth,original_depth)


def test_startup_branch_is_deterministic_and_not_always_selected():
    pose=torch.eye(4);pose[2,3]=.6;k=torch.tensor([[100.,0.,32.],[0.,100.,32.],[0.,0.,1.]])
    vertices=torch.tensor([[-.05,-.05,-.05],[.05,.05,.05]]);rgb=torch.rand(3,64,64)
    starts=[]
    for seed in range(32):
        a=make_plan(rgb,pose,k,vertices,.15,seed,56,8,1.,.5)
        b=make_plan(rgb,pose,k,vertices,.15,seed,56,8,1.,.5)
        assert a.start==b.start and a.end==b.end
        starts.append(a.start)
    assert 0<sum(s==0 for s in starts)<32
