import copy,math
from dataclasses import replace
import torch
from lip.unified.rope3d import rotate_xyz,Gated3DRoPE,enable_cad_rope3d,allowed_migration_keys
from lip.unified.cad_surface import SurfaceRead


def test_rotation_preserves_norm_and_relative_translation():
    torch.manual_seed(71);q=torch.randn(2,4,7,32);k=torch.randn(2,4,9,32)
    a=torch.randn(2,7,3);b=torch.randn(2,9,3);f=torch.tensor([.5,1,2,4,8])
    rq=rotate_xyz(q,a,f);rk=rotate_xyz(k,b,f)
    assert torch.allclose(q.norm(dim=-1),rq.norm(dim=-1),atol=1e-6)
    assert torch.equal(q[...,-2:],rq[...,-2:])
    shift=torch.tensor([.13,-.07,.2])[None,None]
    before=rq@rk.transpose(-1,-2)
    after=rotate_xyz(q,a+shift,f)@rotate_xyz(k,b+shift,f).transpose(-1,-2)
    assert torch.allclose(before,after,atol=9e-5,rtol=2e-5)


def fixture():
    torch.manual_seed(72)
    reader=SurfaceRead(12).eval();patch=torch.randn(1,256,256)
    valid=torch.ones(1,256,dtype=torch.bool);features=torch.randn(1,16,12);geometry=torch.randn(1,16,21)*.1
    available=torch.ones(1,16,dtype=torch.bool)
    extras=(torch.randn(1,256,3)*.2,valid,torch.eye(4)[None],torch.ones(1,256)*.8)
    return reader,(patch,valid,features,geometry,available),extras


def test_zero_gate_preserves_reader_but_can_learn():
    reader,args,extras=fixture();baseline=reader(*args)
    reader.rope3d=Gated3DRoPE();result=reader(*args,*extras)
    assert torch.equal(baseline[0],result[0])
    assert torch.equal(baseline[1]['cad_match_log_prob'],result[1]['cad_match_log_prob'])
    (-result[1]['cad_match_log_prob'][:,:,3].mean()).backward()
    assert reader.rope3d.gain.grad.isfinite().all() and reader.rope3d.gain.grad.abs().sum()>0


def test_missing_depth_zero_confidence_and_dropped_cad_fall_back_exactly():
    reader,args,extras=fixture();baseline=reader(*args);reader.rope3d=Gated3DRoPE()
    reader.rope3d.gain.data.fill_(.4)
    xyz,valid,base,confidence=extras
    for positions,mask,trust in [(xyz*float('nan'),valid&False,confidence),(xyz,valid,confidence*0)]:
        actual=reader(*args,positions,mask,base,trust)
        assert torch.equal(actual[0],baseline[0]) and torch.equal(actual[1]['cad_match_log_prob'],baseline[1]['cad_match_log_prob'])
    args=list(args);args[-1]=args[-1]&False
    a=reader(*args,*extras);args[2]=args[2]*float('nan');args[3]=args[3]*float('nan');b=reader(*args,*extras)
    assert torch.equal(a[0],b[0]) and a[0].isfinite().all()


def test_position_and_confidence_are_detached_and_pose_error_changes_logits():
    torch.manual_seed(73);q=torch.randn(1,4,3,32,requires_grad=True);k=torch.randn(1,4,5,32,requires_grad=True)
    original=q@k.transpose(-1,-2)/math.sqrt(32);rope=Gated3DRoPE();rope.gain.data.fill_(.5)
    xyz=torch.randn(1,3,3,requires_grad=True);cad=torch.randn(1,5,3);valid=torch.ones(1,3,dtype=torch.bool)
    confidence=torch.ones(1,3,requires_grad=True);base=torch.eye(4)[None]
    a=rope(original,q,k,xyz,valid,cad,torch.ones(1,5,dtype=torch.bool),base,confidence,valid)
    shifted=xyz.detach()-torch.tensor([.1,0,0])
    b=rope(original,q,k,shifted,valid,cad,torch.ones(1,5,dtype=torch.bool),base,confidence,valid)
    assert not torch.allclose(a,b)
    a.square().mean().backward()
    assert q.grad.abs().sum()>0 and k.grad.abs().sum()>0
    assert xyz.grad is None and confidence.grad is None


def test_full_model_zero_gate_parity_and_shared_patch_gradient():
    from test_cad_surface import model_sample
    from lip.unified.reconstruction_only import configure_reconstruction_only
    model,obs=model_sample();old={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():before,_=model(obs)
    enable_cad_rope3d(model);configure_reconstruction_only(model)
    with torch.no_grad():after,_=model(obs)
    for key in ('patch_latent','f_predicted','surface_xyz','pose_centered'):
        assert torch.equal(before[key],after[key]),key
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in old.items())
    assert set(model.state_dict())-set(old)=={'cad_surface.rope3d.gain'}
    model.cad_surface.rope3d.gain.data.fill_(.1)
    output,_=model(obs)
    loss=output['surface_xyz'].square().mean()-output['cad_match_log_prob'][:,:,3].mean()
    loss.backward()
    assert model.cad_surface.rope3d.gain.grad.abs().sum()>0
    assert model.core.blocks[0].spatial.q.weight.grad.abs().sum()>0


def test_fullgraph_compile_and_new_parameter_roundtrip():
    from test_cad_surface import model_sample
    model,obs=model_sample();enable_cad_rope3d(model);model.cad_surface.rope3d.gain.data.fill_(.1)
    eager,_=model(obs);model.compiled_frame=torch.compile(model.tensor_frame,backend='aot_eager',fullgraph=True)
    compiled,_=model(obs)
    assert torch.allclose(eager['patch_latent'],compiled['patch_latent'],atol=2e-6,rtol=1e-5)
    compiled['f_predicted'].square().mean().backward()
    assert model.cad_surface.rope3d.gain.grad.isfinite().all()
    other,_=model_sample();enable_cad_rope3d(other);other.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
    assert allowed_migration_keys({})==set()
    assert allowed_migration_keys({'cad_rope3d':{'enabled':True}})=={'cad_surface.rope3d.gain'}
