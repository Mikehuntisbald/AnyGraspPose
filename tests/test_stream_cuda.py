import json
import pytest
import torch
from test_stream_attention import metadata
from lip.models.stream_attention import StreamTemporal


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dual',[False,True])
def test_cuda_fp32_math_and_bf16_reference_and_memory(dual):
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.manual_seed(42);m=StreamTemporal(8,0.).cuda().eval()
    source=torch.randn(1,32,17,256,device='cuda');q=torch.randn(1,32,2 if dual else 1,256,device='cuda')
    meta=[type(x)(*(getattr(x,n).cuda() for n in x.__dataclass_fields__)) for x in metadata(32,1)]
    report={}
    with torch.no_grad():
        for precision in ['fp32','bf16']:
            cache=None;values=[]
            with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                for i in range(32):z,cache=m(source[:,i],q[:,i],meta[i],cache,dual);values.append(z)
                full,_=m.full_reference(source,q,meta,dual)
            actual=torch.stack(values,1);error=(actual-full).abs().max().item();report[precision]=error
            if precision=='fp32':torch.testing.assert_close(actual,full,atol=1e-5,rtol=1e-4)
            else:assert error<.06
        assert cache.kv_bytes==557056
    print(json.dumps(dict(test='cuda_reference',dual=dual,max_abs=report,cache_bytes=cache.kv_bytes)))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_10000_incremental_steps_memory_bounded():
    from lip.engine.stream_state import FrameMeta
    model=StreamTemporal(8,0.).cuda().eval();source=torch.randn(1,17,256,device='cuda');q=torch.randn(1,1,256,device='cuda')
    cache=None;measure=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(10000):
            m=FrameMeta(torch.tensor([i/30],dtype=torch.float64,device='cuda'),torch.tensor([i],device='cuda'),
                torch.zeros(1,dtype=torch.long,device='cuda'),torch.ones(1,17,dtype=torch.bool,device='cuda'),torch.zeros(1,17,device='cuda'))
            z,cache=model(source,q,m,cache)
            if i in (99,999,4999,9999):
                torch.cuda.synchronize();measure.append(dict(step=i+1,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),kv_bytes=cache.kv_bytes))
    assert torch.isfinite(z).all() and all(r['kv_bytes']==557056 for r in measure)
    assert measure[-1]['allocated']-measure[0]['allocated']<2*1024**2
    print(json.dumps(dict(test='10000_temporal_steps',measurements=measure)))
