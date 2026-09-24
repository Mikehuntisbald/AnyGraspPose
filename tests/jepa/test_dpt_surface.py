import copy
import pytest
import torch
from torch.nn import functional as F
from lip.unified.dpt_surface import DPTSurfaceHead, FixedBilinear, enable_dpt_surface, migrate_dpt_surface
from lip.unified.rope3d import enable_cad_rope3d
from lip.unified.reconstruction_only import configure_reconstruction_only, is_pose_parameter


def test_resize_matches_bilinear_forward_and_backward():
    for small, large in [(16,8),(8,16),(16,64),(64,112),(112,224)]:
        x=torch.randn(1,2,small,small,requires_grad=True)
        actual=FixedBilinear(small,large)(x)
        expected=F.interpolate(x,size=(large,large),mode='bilinear',align_corners=False)
        assert torch.allclose(actual,expected,atol=3e-5,rtol=1e-5)
        upstream=torch.randn_like(actual)
        a=torch.autograd.grad(actual,x,upstream,retain_graph=True)[0]
        b=torch.autograd.grad(expected,x,upstream)[0]
        assert torch.allclose(a,b,atol=5e-5,rtol=2e-5)


def test_dense_geometry_reads_all_levels_and_neighboring_patches():
    torch.manual_seed(43);head=DPTSurfaceHead()
    levels=[torch.randn(1,256,256,requires_grad=True) for _ in range(4)]
    result=head(levels,torch.ones(1,256,dtype=torch.bool))
    assert result.shape==(1,5,224,224) and result.isfinite().all()
    result[0,:4,112,112].sum().backward()
    for level in levels:
        norms=level.grad.abs().sum(-1)[0]
        assert (norms>0).sum()>1  # Unlike the old independent patch MLP.
        assert norms[8*16+7]>0 and norms[8*16+8]>0
    valid=torch.ones(1,256,dtype=torch.bool);valid[:,0]=False
    clean=[x.detach().clone() for x in levels]
    altered=[x.clone() for x in clean]
    for x in altered:x[:,0]=float('nan')
    assert torch.equal(head(clean,valid),head(altered,valid))


def setup_model():
    from test_cad_surface import model_sample
    model,obs=model_sample()
    old={k:v.clone() for k,v in model.state_dict().items()}
    enable_cad_rope3d(model);enable_dpt_surface(model,{'kind':'dpt','width':64})
    replaced=migrate_dpt_surface(model,old)
    configure_reconstruction_only(model)
    return model,obs,old,replaced


def test_migration_preserves_shared_tensors_and_rejects_missing_trunk():
    model,obs,old,replaced=setup_model()
    assert replaced and all(k.startswith('surface_head.') for k in replaced)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in old.items() if k not in replaced)
    assert 'surface_head.3.weight' not in model.state_dict()
    bad=dict(old);del bad['core.blocks.0.spatial.q.weight']
    with pytest.raises(ValueError,match='state mismatch'):migrate_dpt_surface(model,bad)


def test_unified_gradients_and_geometry_output_contract():
    model,obs,old,replaced=setup_model()
    output,_=model(obs)
    assert torch.allclose(output['surface_depth_m'],obs.base[:,2,3,None,None,None]+
                          output['surface_depth_residual']*obs.diameter[:,None,None,None])
    loss=output['surface_xyz'].square().mean()+output['surface_depth_m'].square().mean()
    loss=loss+F.binary_cross_entropy_with_logits(output['geometry_valid_logits'],torch.ones_like(output['geometry_valid_logits']))
    loss.backward()
    assert model.cad_surface.rope3d.gain.grad.abs().sum()>0
    for i in range(4):
        assert model.core.blocks[i].spatial.q.weight.grad.abs().sum()>0
        assert model.surface_head.projects[i].weight.grad.abs().sum()>0
    assert all(p.grad is None for n,p in model.named_parameters() if is_pose_parameter(n))


def test_dpt_optimizer_group_and_checkpoint_roundtrip():
    from lip.unified.optimized_training import make_optimizer
    model,obs,_,_=setup_model()
    config={'training':{'learning_rates':{'new':1e-4,'predictor':1e-5,'pose':3e-6},'weight_decay':.01}}
    optimizer=make_optimizer(model,config,fused=False)
    owners={n:g['category'] for g in optimizer.param_groups for n in g['names']}
    assert all(owners[n]=='new' for n,_ in model.surface_head.named_parameters(prefix='surface_head'))
    assert not any(is_pose_parameter(n) for n in owners)
    other,_,_,_=setup_model();other.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())


def test_dpt_and_rope_fullgraph_forward_backward():
    model,obs,_,_=setup_model();model.cad_surface.rope3d.gain.data.fill_(.1)
    expected,_=model(obs)
    model.compiled_frame=torch.compile(model.tensor_frame,backend='aot_eager',fullgraph=True)
    actual,_=model(obs)
    for key in ('surface_xyz','surface_depth_m','geometry_valid_logits','patch_latent'):
        assert torch.allclose(actual[key],expected[key],atol=3e-6,rtol=1e-5),key
    actual['surface_xyz'].square().mean().backward()
    assert model.surface_head.output[-1].weight.grad.isfinite().all()
