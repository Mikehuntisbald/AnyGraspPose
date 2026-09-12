import copy
import json
import os
from pathlib import Path
import pytest
import torch
from lip.engine.stream_state import FrameMeta,TemporalCache,RingCache
from lip.models.stream_attention import StreamTemporal

torch.set_num_threads(2)


def metadata(frames,batch=2,reset=False):
    result=[]
    for i in range(frames):
        valid=torch.ones(batch,17,dtype=torch.bool)
        if batch>1 and i>=max(1,frames//2):valid[1]=False
        tag=torch.arange(batch,dtype=torch.long)+(100 if reset and i>=9 else 0)
        result.append(FrameMeta(torch.tensor([1789100000.+i*.027+(i%3)*.002]*batch,dtype=torch.float64),
            torch.full((batch,),i,dtype=torch.long),tag,valid,torch.linspace(-2.,0.,17).expand(batch,-1)))
    return result


@pytest.mark.parametrize('frames',[1,2,8,9,32])
@pytest.mark.parametrize('dual',[False,True])
def test_cached_reference_after_eviction(frames,dual):
    torch.manual_seed(43);m=StreamTemporal(8,0.).eval()
    # Nonzero bias makes float64 timestamp semantics part of the comparison.
    torch.nn.init.normal_(m.time_bias[-1].weight,std=.03)
    source=torch.randn(2,frames,17,256);q=torch.randn(2,frames,2 if dual else 1,256)
    meta=metadata(frames,reset=True);cache=None;outputs=[]
    with torch.no_grad():
        for i in range(frames):
            z,cache=m(source[:,i],q[:,i],meta[i],cache,dual);outputs.append(z)
            assert len(cache.metadata)<=8
            assert all(b.key.shape==(2,8,17,32) for layer in cache.layers for b in layer)
        ref,kv=m.full_reference(source,q,meta,dual)
    actual=torch.stack(outputs,1);error=(actual-ref).abs().max().item()
    print(json.dumps(dict(test='cache_reference',frames=frames,dual=dual,max_abs=error)))
    torch.testing.assert_close(actual,ref,atol=1e-5,rtol=1e-4)


def test_functional_cache_gradients_match_reference_and_reach_past():
    torch.manual_seed(11);a=StreamTemporal(2,0.).eval();b=copy.deepcopy(a)
    x=torch.randn(1,4,17,256,requires_grad=True);y=x.detach().clone().requires_grad_(True)
    query=torch.randn(1,4,1,256);meta=metadata(4,1);cache=None
    for i in range(4):z,cache=a(x[:,i],query[:,i],meta[i],cache)
    loss=z.square().mean()+z[...,0].sum();loss.backward()
    ref,_=b.full_reference(y,query,meta);(ref[:,-1].square().mean()+ref[:,-1,...,0].sum()).backward()
    assert x.grad[:,0].abs().sum()>0 # multi-layer state carries information older than W
    torch.testing.assert_close(x.grad,y.grad,atol=1e-5,rtol=1e-4)
    worst=0.
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb and pa.grad is not None and pb.grad is not None
        torch.testing.assert_close(pa.grad,pb.grad,atol=2e-5,rtol=2e-4)
        worst=max(worst,(pa.grad-pb.grad).abs().max().item())
    print(json.dumps(dict(test='cache_gradient_reference',max_abs=worst)))


def test_queries_never_enter_source_cache_and_history_not_reencoded():
    m=StreamTemporal(8,0.).eval();meta=metadata(2,1);s=torch.randn(1,17,256)
    with torch.no_grad():
        _,a=m(s,torch.zeros(1,1,256),meta[0]);_,b=m(s,torch.randn(1,2,256),meta[0],object_readout=True)
        snapshots=[x[0].key.clone() for x in a.layers]
        _,c=m(s,torch.randn(1,1,256),meta[1],a)
    for i in range(4):
        assert torch.equal(a.layers[i][0].key,b.layers[i][0].key)
        assert torch.equal(a.layers[i][0].value,b.layers[i][0].value)
        assert torch.equal(c.layers[i][0].key,snapshots[i])
    assert c.metadata[0].timestamp.dtype==torch.float64


def test_all_invalid_padding_sources_are_zero_and_cannot_pollute_real_lane():
    m=StreamTemporal(8,0.).eval();meta=metadata(2,2)
    empty=FrameMeta(meta[0].timestamp,meta[0].frame_id,meta[0].stream_tag,torch.zeros(2,17,dtype=torch.bool),meta[0].role_bias)
    with torch.no_grad():
        z,cache=m(torch.full((2,17,256),float('nan')),torch.randn(2,1,256),empty)
        assert torch.equal(z,torch.zeros_like(z))
        assert all(torch.count_nonzero(b.key)==0 and torch.count_nonzero(b.value)==0 for layer in cache.layers for b in layer)
        s=torch.randn(2,17,256);q=torch.randn(2,1,256)
        after,_=m(s,q,meta[1],cache);fresh,_=m(s,q,meta[1])
    torch.testing.assert_close(after,fresh,atol=1e-5,rtol=1e-4)


def test_future_and_other_streams_cannot_affect_past():
    m=StreamTemporal(8,0.).eval();meta=metadata(9,1);s=torch.randn(1,9,17,256);q=torch.randn(1,9,2,256)
    with torch.no_grad():
        a,_=m.full_reference(s,q,meta,True)
        s[:,5:]*=100;q[:,5:]*=-50
        changed=meta[:5]+[FrameMeta(x.timestamp+50,x.frame_id,x.stream_tag,x.key_valid,x.role_bias) for x in meta[5:]]
        b,_=m.full_reference(s,q,changed,True)
    assert torch.equal(a[:,:5],b[:,:5])


def test_ring_is_equivalent_and_bounded_for_10000_appends():
    from lip.engine.stream_state import KVBlock
    ring=RingCache.empty(8);meta=metadata(1,1)[0]
    for i in range(10000):
        blocks=[KVBlock(torch.full((1,8,17,32),float(i),dtype=torch.bfloat16),torch.zeros(1,8,17,32,dtype=torch.bfloat16)) for _ in range(4)]
        ring=ring.append(blocks,meta)
        if i in (0,7,8,9999):
            c=ring.functional();assert c.kv_bytes==min(i+1,8)*69632
            assert c.layers[0][-1].key[0,0,0,0]==torch.tensor(float(i),dtype=torch.bfloat16)
    assert ring.kv_bytes==557056 and ring.count==8
    # Actual temporal outputs using ring ordering agree with functional updates.
    m=StreamTemporal(3,0.).eval();ring=RingCache.empty(3);cache=None
    for i,meta in enumerate(metadata(9,1)):
        s=torch.randn(1,17,256);q=torch.randn(1,1,256)
        with torch.no_grad():
            a,cache=m(s,q,meta,cache);b,next_cache=m(s,q,meta,ring.functional())
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
        ring=ring.append([l[-1] for l in next_cache.layers],meta)
