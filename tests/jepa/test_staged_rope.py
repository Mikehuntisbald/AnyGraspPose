from dataclasses import replace
from types import SimpleNamespace
import torch
from lip.unified.staged_rope import enable_staged_rope,route_surface,measured_depth_stats,coarse_surface_loss,preserve_adam


PLAN=dict(visible_threshold=.7,measured_min_mass=.5,measured_max_std_d=.05,recovered_threshold=.5,recovered_support_threshold=.7,recovered_max_trust=.5)


def routing_inputs():
    xyz=torch.ones(1,256,3,requires_grad=True)
    valid=torch.ones(1,256,dtype=torch.bool)
    visibility=torch.full((1,256),.2);visibility[:,0]=.9
    stats=torch.zeros(1,256,2);stats[...,0]=1
    base=torch.eye(4)[None];base[:,:3,3]=torch.tensor([.1,-.2,1.])
    diameter=torch.tensor([.5]);rays=torch.zeros(1,3,224,224);rays[:,2]=1
    coarse=torch.zeros(1,5,224,224);coarse[:,3]=.25;coarse[:,4]=5
    return xyz,valid,visibility,stats,base,diameter,rays,valid,coarse.requires_grad_()


def test_measured_priority_recovered_camera_lift_and_no_positional_gradients():
    xyz,depth_ok,p,stats,base,d,rays,valid,coarse=routing_inputs()
    position,chosen,trust,info=route_surface(xyz,depth_ok,p,stats,base,d,rays,valid,PLAN,coarse,torch.ones_like(p))
    assert info['rope_measured_mask'][0,0] and not info['rope_recovered_mask'][0,0]
    assert torch.equal(position[:,0],xyz.detach()[:,0])
    assert info['rope_recovered_mask'][0,1:].all()
    assert torch.allclose(position[:,1],torch.tensor([[-.2,.4,.25]]),atol=1e-6)
    assert chosen.all() and trust[:,1:].max()<=.5
    assert not position.requires_grad and not trust.requires_grad
    _,_,_,without_support=route_surface(xyz,depth_ok,p,stats,base,d,rays,valid,PLAN,coarse,torch.zeros_like(p))
    assert not without_support['rope_recovered_mask'].any()
    missing=depth_ok.clone();missing[:,0]=False
    _,_,missing_trust,missing_info=route_surface(xyz,missing,p,stats,base,d,rays,valid,PLAN,coarse,torch.ones_like(p))
    assert missing_info['rope_recovered_mask'][0,0] and missing_trust[0,0]>.4
    other=coarse.detach().clone();other[:,:3]=1000
    assert torch.equal(position,route_surface(xyz,depth_ok,p,stats,base,d,rays,valid,PLAN,other,torch.ones_like(p))[0])
    other[:,3]+=.5
    moved=route_surface(xyz,depth_ok,p,stats,base,d,rays,valid,PLAN,other,torch.ones_like(p))[0]
    assert torch.allclose(moved[:,1,2]-position[:,1,2],torch.tensor([.5]),atol=1e-6)


def test_early_read_has_no_recovery_and_bad_measurements_are_not_trusted():
    xyz,depth_ok,p,stats,base,d,rays,valid,coarse=routing_inputs()
    _,chosen,_,info=route_surface(xyz,depth_ok,p,stats,base,d,rays,valid,PLAN)
    assert chosen.sum()==1 and not info['rope_recovered_mask'].any()
    stats[:,0,1]=.1
    _,chosen,trust,_=route_surface(xyz,depth_ok,p,stats,base,d,rays,valid,PLAN)
    assert not chosen.any() and not trust.any()
    bad=coarse.detach().clone();bad[:,3]=float('nan')
    positions,chosen,trust,info=route_surface(xyz,depth_ok,p,stats,base,d,rays,valid,PLAN,bad,torch.ones_like(p))
    assert not chosen.any() and not trust.any() and positions.isfinite().all()
    assert info['rope_fallback_mask'].all()


def test_depth_quality_uses_actual_measurement_mass_and_spread():
    depth=torch.ones(1,1,224,224);depth[:,:,0:14,0:14]=0
    depth[:,:,14:21,0:14]=1.1
    base=torch.eye(4)[None];base[:,2,3]=1
    stats=measured_depth_stats(depth,base,torch.tensor([.2]))
    assert stats[0,0,0]==0 and stats[0,1,0]==1
    assert torch.allclose(stats[0,16,1],torch.tensor(.25),atol=1e-5)


def sample():
    from test_dpt_surface import setup_model
    model,obs,_,_=setup_model()
    with torch.no_grad():
        model.visibility[-1].weight.zero_();model.visibility[-1].bias.fill_(2)
        model.surface_head.output[-1].bias[4]=2
        model.core.support.weight.zero_();model.core.support.bias.fill_(2)
    state={k:v.clone() for k,v in model.state_dict().items()}
    enable_staged_rope(model,PLAN)
    rays=torch.zeros(1,3,224,224);rays[:,2]=1
    quality=torch.zeros(1,256,2);quality[...,0]=1
    depth=obs.depth_valid.clone();depth[:,128:]=False
    return model,replace(obs,crop_rays=rays,rope_depth_stats=quality,depth_valid=depth),state


def test_shared_decoder_weights_no_new_parameters_and_no_cyclic_dependency():
    model,obs,before=sample()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and set(before)==set(model.state_dict())
    events=[]
    hooks=[block.register_forward_hook(lambda module,args,out,i=i:events.append('block'+str(i+1))) for i,block in enumerate(model.core.blocks)]
    hooks.append(model.surface_head.register_forward_hook(lambda module,args,out:events.append('dpt')))
    with torch.no_grad():first,_=model(obs)
    for hook in hooks:hook.remove()
    assert events==['block1','block2','block3','block4','dpt','block4','dpt']
    assert first['rope_measured_mask'].any() and first['rope_recovered_mask'].any()
    assert not (first['rope_measured_mask']&first['rope_recovered_mask']).any()
    assert first['coarse_surface'].shape==(1,5,224,224)


def test_coarse_geometry_loss_trains_shared_decoder_and_prefix():
    model,obs,_=sample();out,_=model(obs)
    shape=out['coarse_surface'].shape;mask=torch.ones(shape[0],1,shape[2],shape[3],dtype=torch.bool)
    target=SimpleNamespace(surface_xyz=torch.zeros_like(out['surface_xyz']),surface_depth_residual=torch.zeros_like(out['surface_depth_residual']),
        geometry_real_weight=mask,geometry_proxy_weight=mask&False,geometry_valid_label=mask.float(),
        camera_rotation=torch.eye(3)[None],camera_translation_d=torch.tensor([[0.,0.,1.]]),
        camera_rays=obs.crop_rays,base_depth_d=torch.tensor([1.]))
    loss,parts=coarse_surface_loss(out,target,dict(surface_xyz=.5,surface_depth=.5,camera_consistency=.25,geometry_validity=.1,cad_feature=.5))
    loss.backward()
    assert model.surface_head.output[-1].weight.grad.abs().sum()>0
    assert model.core.blocks[2].spatial.q.weight.grad.abs().sum()>0
    assert model.core.blocks[3].spatial.q.weight.grad.abs().sum()>0
    assert abs(sum(float(parts['rope_'+n+'_fraction']) for n in ('measured','recovered','fallback'))-1)<1e-6


def test_fullgraph_compile_preserves_staged_routing():
    model,obs,_=sample();expected,_=model(obs)
    model.compiled_frame=torch.compile(model.tensor_frame,backend='aot_eager',fullgraph=True)
    actual,_=model(obs)
    for key in ('coarse_surface','surface_xyz','patch_latent'):
        assert torch.allclose(expected[key],actual[key],atol=3e-6,rtol=1e-5),key
    (actual['coarse_surface'].square().mean()+actual['surface_xyz'].square().mean()).backward()
    assert model.cad_surface.rope3d.gain.grad.isfinite().all()


def test_preserve_adam_moments_step_and_parameter_groups():
    from lip.unified.horizon_resume import exact
    a=torch.nn.Linear(3,2);b=torch.nn.Linear(3,2)
    def optimizer(m):return torch.optim.AdamW([dict(params=list(m.parameters()),names=['weight','bias'],lr=1e-5)])
    first=optimizer(a);a(torch.ones(1,3)).sum().backward();first.step()
    second=optimizer(b);source=dict(optimizer=first.state_dict());preserve_adam(second,source)
    assert exact(second.state_dict(),source['optimizer'])
