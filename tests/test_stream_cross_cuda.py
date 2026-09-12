import json
import pytest
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from lip.engine.stream_state import FrameMeta,CrossCache,TemporalCache
from lip.models.stream_tracker import StreamTracker


def meta(i,device):
    return FrameMeta(torch.tensor([1789180000.+i*.034],device=device,dtype=torch.float64),
                     torch.tensor([i],device=device),torch.tensor([0],device=device),
                     torch.ones(1,17,device=device,dtype=torch.bool),torch.linspace(-2.,0.,17,device=device)[None])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_combined_source_and_context_cached_reference_cuda(precision):
    torch.set_num_threads(2);torch.manual_seed(7)
    m=StreamTracker('stream_dual_cross').cuda().eval();torch.nn.init.normal_(m.readout.cross_attn.out_proj.weight,std=.03)
    x=torch.randn(1,12,17,256,device='cuda');metadata=[meta(i,'cuda') for i in range(12)]
    query=m.readout.query+m.token_type[2];cache=CrossCache(TemporalCache());outputs=[]
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
        for i in range(12):
            z,source=m.temporal(x[:,i],query,metadata[i],cache.source,True)
            out,contexts=m.readout(z,metadata[i],cache.contexts);cache=CrossCache(source,contexts);outputs.append(out['latent'])
        z,_=m.temporal.full_reference(x,query[:,None].expand(-1,12,-1,-1),metadata,True)
        reference,_,_=m.readout.full_reference(z,metadata)
    actual=torch.stack(outputs,1);error=float((actual-reference).abs().max());print('combined_cross_reference',precision,error)
    tolerance=1e-5 if precision=='fp32' else .04
    torch.testing.assert_close(actual,reference,atol=tolerance,rtol=tolerance)
    assert cache.kv_bytes==(565248 if precision=='bf16' else 1130496)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cross_cuda_10000_updates_memory_bounded():
    torch.set_num_threads(2);m=StreamTracker('stream_dual_cross').cuda().eval()
    x=torch.randn(1,17,256,device='cuda');query=m.readout.query+m.token_type[2]
    cache=CrossCache(TemporalCache());samples=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(10000):
            frame=meta(i,'cuda');z,source=m.temporal(x,query,frame,cache.source,True)
            out,contexts=m.readout(z,frame,cache.contexts);cache=CrossCache(source,contexts)
            if i+1 in (100,1000,5000,10000):
                torch.cuda.synchronize();samples.append(dict(step=i+1,allocated=torch.cuda.memory_allocated(),kv_bytes=cache.kv_bytes))
    assert all(r['kv_bytes']==565248 for r in samples)
    assert max(r['allocated'] for r in samples)-min(r['allocated'] for r in samples)<2*1024**2
    assert all(b.key.grad_fn is None for b in cache.contexts)
    print('cross_memory',json.dumps(samples))
