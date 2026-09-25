from types import SimpleNamespace
import pytest
import torch
from torch import nn
from lip.unified.cad_surface import CADSurfaceTracker


@pytest.mark.parametrize('amp',[False,True])
def test_detached_rope_preview_preserves_main_projection_and_visibility_gradients(amp):
    torch.manual_seed(42)
    model=SimpleNamespace(core=SimpleNamespace(src_proj=nn.Linear(8,4)),
        visibility=nn.Sequential(nn.LayerNorm(4),nn.Linear(4,1)),cad_rope3d_enabled=True)
    mid=torch.randn(1,3,4,requires_grad=True);last=torch.randn_like(mid,requires_grad=True)
    obs=SimpleNamespace(mid=mid,last=last,cad_surface_features=torch.randn(1,4,8),
        cad_surface_geometry=torch.randn(1,4,21),cad_surface_valid=torch.ones(1,4,dtype=torch.bool),
        cad_valid=torch.ones(1,3,dtype=torch.bool),object_xyz=torch.randn(1,3,3),
        depth_valid=torch.ones(1,3,dtype=torch.bool),base=torch.eye(4)[None],state=torch.zeros(1,24))
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        inputs=CADSurfaceTracker.extra_frame_inputs(model,obs)[-1]
        preview=inputs[-1]
        projected=model.core.src_proj(torch.cat((mid,last),-1))
        visible=model.visibility(projected).float().squeeze(-1).sigmoid()
        assert not preview.requires_grad
        torch.testing.assert_close(preview,visible.detach(),rtol=0,atol=0)
        loss=projected.float().square().mean()+(visible-.2).square().mean()
    loss.backward()
    for p in (*model.core.src_proj.parameters(),*model.visibility.parameters()):
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0
    assert mid.grad is not None and last.grad is not None
