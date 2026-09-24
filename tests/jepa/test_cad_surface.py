from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import hashlib
import pytest
import torch
from torch import nn
from lip.unified.cad_surface import CADSurfaceTracker,migrate_surface,SurfaceRead,encode_surface
from lip.unified.cad_surface_cache import surface_tokens
from lip.unified.cad_surface_targets import orbit_distance,surface_targets,surface_correspondence_loss
from lip.unified.reconstruction_only import configure_reconstruction_only
from test_recovered_relation import sample


def model_sample():
    _,parent,obs=sample();point=nn.Identity();point.feature_dim=12
    model=CADSurfaceTracker(nn.Identity(),point).eval();migrate_surface(model,parent.state_dict())
    geometry=torch.randn(1,16,21)*.1;geometry[:,:,11]=5.;geometry[:,:,15:17]=torch.rand(1,16,2)
    obs=replace(obs,cad_surface_features=torch.randn(1,16,12),cad_surface_geometry=geometry,cad_surface_valid=torch.ones(1,16,dtype=torch.bool))
    return model,obs


def test_actual_read_and_correspondence_reach_shared_jepa_without_pose_bypass():
    model,obs=model_sample();configure_reconstruction_only(model)
    out,_=model(obs)
    truth=torch.zeros_like(out['cad_match_log_prob']);truth[:,:,3]=1
    t=SimpleNamespace(cad_surface_target_real=truth,cad_surface_target_proxy=truth*0,
                      cad_surface_weight_real=torch.ones(1,256),cad_surface_weight_proxy=torch.zeros(1,256))
    loss,_=surface_correspondence_loss(out,t)
    grad=torch.autograd.grad(loss,(model.cad_surface.q.weight,model.cad_surface.k.weight,
                                  model.core.blocks[0].spatial.q.weight),retain_graph=True)
    assert all(g.isfinite().all() and g.abs().sum()>0 for g in grad)
    geometry_grad=torch.autograd.grad(out['surface_xyz'].square().mean(),model.cad_surface.v.weight)[0]
    assert geometry_grad.abs().sum()>0


def test_dropped_cad_cannot_leak_local_features_even_when_valid_mask_is_stale():
    model,obs=model_sample();obs=replace(obs,cad_valid=obs.cad_valid&False)
    with torch.no_grad():
        a,_=model(obs)
        b,_=model(replace(obs,cad_surface_features=obs.cad_surface_features+1e6,cad_surface_geometry=obs.cad_surface_geometry-1e6))
    for key in ('patch_latent','surface_xyz','f_predicted','pose_centered'):
        assert torch.equal(a[key],b[key]),key
    assert not a['cad_surface_available'].any()


def test_local_cad_changes_recovery_but_never_written_real_observations():
    model,obs=model_sample()
    with torch.no_grad():
        a,ma=model(obs);b,mb=model(replace(obs,cad_surface_features=-obs.cad_surface_features))
    assert not torch.equal(a['patch_latent'],b['patch_latent']) and not torch.equal(a['surface_xyz'],b['surface_xyz'])
    for name in ('objects','contexts'):
        for x,y in zip(getattr(ma,name),getattr(mb,name)):
            for field in x.__dataclass_fields__:assert torch.equal(getattr(x,field),getattr(y,field))


def cad_fixture():
    torch.manual_seed(13)
    return dict(coord=torch.randn(64,3)*.15,normal=torch.nn.functional.normalize(torch.randn(64,3),dim=-1),
                color=torch.rand(64,3),features=torch.randn(64,12),key='parent-a',receipt={'sha256':'sha-a'},
                _asset_files=[Path('/models/test/textured.obj')])


def test_cache_is_deterministic_local_and_bound_to_parent(tmp_path):
    cad=cad_fixture();before=cad['features'].clone();a=surface_tokens(cad,tmp_path,16)
    assert len(a['coord'])==16 and torch.equal(cad['features'],before)
    other=cad_fixture();b=surface_tokens(other,tmp_path,16)
    assert all(torch.equal(a[k],b[k]) for k in ('coord','features','normal','color','indices'))
    other['key']='parent-b';other['receipt']={'sha256':'sha-b'};other['features']+=1
    c=surface_tokens(other,tmp_path,16)
    assert not torch.equal(b['features'],c['features']) and len(list(tmp_path.glob('*.pt')))==2


def test_discrete_and_continuous_symmetries_do_not_force_false_point_identity():
    anchor=torch.tensor([[1.,0.,0.]])
    opposite=torch.tensor([[-1.,0.,0.]])
    half=torch.diag(torch.tensor([-1.,-1.,1.,1.])).flatten().tolist()
    assert orbit_distance(opposite,anchor,{},torch.zeros(3),1.).item()==4
    assert orbit_distance(opposite,anchor,{'symmetries_discrete':[half]},torch.zeros(3),1.).item()==0
    meta={'symmetries_continuous':[dict(axis=[0,0,1],offset=[0,0,0])]}
    assert orbit_distance(torch.tensor([[0.,1.,0.]]),anchor,meta,torch.zeros(3),1.).item()==0
    assert orbit_distance(torch.tensor([[0.,1.,1.]]),anchor,meta,torch.zeros(3),1.).item()==1


def test_dynamic_camera_geometry_and_missing_depth_are_explicit(tmp_path):
    cad=cad_fixture();pose=torch.eye(4);pose[2,3]=1.
    scene=SimpleNamespace(cad=cad,pose=pose,diameter=.2,k_crop=torch.tensor([[400.,0.,112.],[0.,400.,112.],[0.,0.,1.]]))
    model=SimpleNamespace(cad_surface_cache=tmp_path,cad_surface_count=16)
    _,g,v=encode_surface(model,[scene],torch.zeros(1,1,224,224),torch.tensor([True]))
    assert not g[:,:,17:19].any() and v.any()
    scene.pose=pose.clone();scene.pose[0,3]=.05
    _,changed,_=encode_surface(model,[scene],torch.ones(1,1,224,224),torch.tensor([True]))
    assert not torch.equal(g[:,:,9:12],changed[:,:,9:12]) and not torch.equal(g[:,:,15:17],changed[:,:,15:17])
    assert torch.equal(g[:,:,:9],changed[:,:,:9])


def test_correspondence_labels_use_surface_distribution_not_patch_mean(tmp_path):
    from lip.unified.features import TeacherTargets
    cad=cad_fixture();bank=surface_tokens(cad,tmp_path,16)
    xyz=torch.zeros(1,3,224,224)
    xyz[:,:,:,:]=bank['coord'][0][:,None,None]
    xyz[:,:,:,7:14]=bank['coord'][8][:,None,None]
    mask=torch.ones(1,1,224,224,dtype=torch.bool)
    feature=torch.zeros(1,256,384);weight=torch.ones(1,256)
    target=TeacherTargets(feature,feature,feature,feature,weight*0,weight,weight*0,weight*0,weight*0,
        weight,weight,torch.zeros(1,3,224,224),torch.zeros(1,3,224,224),xyz,mask.float(),mask.float(),
        mask,mask,~mask,mask.float(),mask,~mask)
    scene=SimpleNamespace(cad=cad,center=torch.zeros(3),diameter=.2)
    model=SimpleNamespace(cad_surface_cache=tmp_path,cad_surface_count=16,cad_surface_symmetry={'test':{}})
    out=surface_targets(model,[scene],target)
    row=out.cad_surface_target_real[0,0]
    assert row[0]>.25 and row[8]>.25 and torch.allclose(row.sum(),torch.tensor(1.))
    assert not out.cad_surface_target_proxy.any()
    assert torch.equal(target.surface_xyz,xyz)


def test_no_cad_or_no_targets_has_finite_zero_correspondence_gradient():
    logits=torch.randn(1,256,17,requires_grad=True)
    output=dict(cad_match_log_prob=logits.log_softmax(-1),cad_surface_available=torch.zeros(1,16,dtype=torch.bool))
    truth=torch.ones_like(logits)/17;weight=torch.ones(1,256)
    target=SimpleNamespace(cad_surface_target_real=truth,cad_surface_target_proxy=truth,
                          cad_surface_weight_real=weight,cad_surface_weight_proxy=weight)
    loss,_=surface_correspondence_loss(output,target);loss.backward()
    assert loss==0 and not logits.grad.any() and logits.grad.isfinite().all()


def test_fullgraph_aot_forward_and_backward_preserve_local_cad_read():
    model,obs=model_sample();configure_reconstruction_only(model)
    eager,_=model(obs)
    model.compiled_frame=torch.compile(model.tensor_frame,backend='aot_eager',fullgraph=True)
    compiled,_=model(obs)
    for key in ('patch_latent','surface_xyz','cad_match_log_prob'):
        assert torch.allclose(eager[key],compiled[key],atol=2e-6,rtol=1e-5)
    compiled['surface_xyz'].square().mean().backward()
    assert model.cad_surface.q.weight.grad.abs().sum()>0


def test_cache_corruption_is_rejected(tmp_path):
    import json
    cad=cad_fixture();surface_tokens(cad,tmp_path,16)
    receipt=next(tmp_path.glob('*.json'));meta=json.loads(receipt.read_text());meta['sha256']='invalid'
    receipt.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='identity'):surface_tokens(cad_fixture(),tmp_path,16)


def test_compiled_all_cad_dropped_diagnostic_has_no_nan_backward():
    model,obs=model_sample();configure_reconstruction_only(model)
    obs=replace(obs,cad_valid=obs.cad_valid&False)
    model.compiled_frame=torch.compile(model.tensor_frame,backend='aot_eager',fullgraph=True)
    out,_=model(obs);out['surface_xyz'].square().mean().backward()
    assert out['cad_surface_update_rms']==0 and not out['cad_surface_update_rms'].requires_grad
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
