import pytest
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from lip.models.stream_attention import StreamLayer


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_frozen_qkv_trainable_bias_backward_matches_math_reference():
    torch.manual_seed(3);layer=StreamLayer(0.).cuda().eval()
    for p in layer.parameters():p.requires_grad_(False)
    q=torch.randn(2,8,19,32,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(2,8,136,32,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    x=torch.randn(2,19,256,device='cuda',dtype=torch.bfloat16)
    bias=torch.randn(2,8,19,136,device='cuda',requires_grad=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):actual=layer.residual(x,q,k,v,bias)
    actual.float().square().mean().backward();grad=bias.grad.clone()
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    bias.grad=None
    with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16):expected=layer.residual(x,q,k,v,bias)
    expected.float().square().mean().backward()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(grad,bias.grad,rtol=0,atol=0)
