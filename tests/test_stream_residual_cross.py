import copy
import pytest
import torch
from lip.models.stream_tracker import StreamTracker
from lip.models.stream_cross_readout import StreamResidualCrossReadout
from lip.engine.stream_checkpoint import migrate
from lip.engine.stream_config import load_stream_config,optimizer_and_scheduler
from lip.engine.stream_state import cache_contract_for
from test_stream_attention import metadata


def migrated(tmp_path,device='cpu'):
    torch.manual_seed(91)
    parent=StreamTracker('stream_dual',memory_frames=8).to(device).eval()
    parent.readout.context_projection.weight.data.normal_(std=.03)
    parent.readout.gate[-1].bias.data.fill_(2.)
    parent.head[-1].weight.data.normal_(std=.001)
    path=tmp_path/'parent.pt'
    torch.save(dict(model=parent.state_dict(),architecture_id='stream_dual',new_stage_step=8800,split_hash='s',mesh_hash='m'),path)
    child=StreamTracker('stream_dual_cross_residual',memory_frames=8).to(device).eval()
    report=migrate(child,path)
    assert not report['source_skipped']
    for name,p in parent.state_dict().items():torch.testing.assert_close(child.state_dict()[name],p,rtol=0,atol=0)
    return parent,child


def test_residual_migration_exact_parent_function_and_contract(tmp_path):
    parent,child=migrated(tmp_path)
    assert cache_contract_for(parent.architecture_id)!=cache_contract_for(child.architecture_id)
    z=torch.randn(2,256).reshape(1,2,256)
    out,blocks=child.readout(z,metadata(1,1)[0])
    torch.testing.assert_close(out['latent'],parent.readout(z)['latent'],rtol=0,atol=0)
    assert out['cross_attention_output'].count_nonzero()==0


def test_residual_cached_reference_mask_and_eviction():
    torch.manual_seed(5);model=StreamResidualCrossReadout(memory_frames=8)
    model.context_projection.weight.data.normal_(std=.03)
    model.cross_attn.out_proj.weight.data.normal_(std=.03)
    meta=metadata(12,2);z=torch.randn(2,12,2,256);contexts=();actual=[]
    for i,m in enumerate(meta):
        out,contexts=model(z[:,i],m,contexts);actual.append(out['latent'])
    reference,_,_=model.full_reference(z,meta)
    torch.testing.assert_close(torch.stack(actual,1),reference,atol=2e-6,rtol=1e-5)
    assert len(contexts)==8


def test_parent_weights_frozen_while_new_branch_learns(tmp_path):
    parent,child=migrated(tmp_path)
    c=load_stream_config('configs/stream_lip_v2_residual_8800.yaml')
    opt,scheduler=optimizer_and_scheduler(child,c)
    assert all(g['lr']==0 for g in opt.param_groups if g['category'] in ('loaded','rgb'))
    initial={n:p.detach().clone() for n,p in child.named_parameters()}
    z=torch.randn(1,2,256);meta=metadata(1,1)[0]
    for _ in range(5):
        opt.zero_grad(set_to_none=True);out,_=child.readout(z,meta)
        child.head(out['latent']).square().mean().backward();opt.step();scheduler.step()
    for n,p in parent.named_parameters():torch.testing.assert_close(dict(child.named_parameters())[n],initial[n],rtol=0,atol=0)
    assert not torch.equal(child.readout.cross_attn.out_proj.weight,initial['readout.cross_attn.out_proj.weight'])
    assert child.readout.cross_attn.in_proj_weight.grad.abs().sum()>0
    assert child.readout.cross_gate[-1].weight.grad.abs().sum()>0


@pytest.mark.cuda
@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_residual_cuda_full_model_parent_equivalence(tmp_path,precision):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from test_stream_geometry import fixture
    from lip.geometry.renderer import Renderer
    mesh,t,k=fixture();parent,child=migrated(tmp_path,'cuda')
    states=[m.initialize(t,mesh,k,'s',0.,image_shape=(64,64)) for m in [parent,child]]
    renderer=Renderer('cuda');rgb=torch.rand(3,64,64,device='cuda');depth=torch.full((1,64,64),.6,device='cuda')
    for i in range(1,11):
        proposals=[]
        for j,m in enumerate([parent,child]):
            p,n=m.step(rgb,depth,.03*i,states[j],renderer=renderer,precision=precision,image_size=32)
            assert p['status']=='ok';states[j]=m.commit(p,n);proposals.append(p)
        for key in ['pose_centered','latent','delta_rotvec','delta_center_norm']:
            torch.testing.assert_close(proposals[0][key],proposals[1][key],rtol=0,atol=0)
