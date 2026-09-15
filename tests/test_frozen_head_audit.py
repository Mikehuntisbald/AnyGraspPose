import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('frozen_head_audit',Path(__file__).resolve().parents[1]/'tools/audit_frozen_pose_head.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_analytic_gelu_head_jacobian_matches_autograd():
    torch.manual_seed(4)
    weights=[torch.randn(16,12,dtype=torch.float64)*.2,torch.randn(16,dtype=torch.float64),torch.randn(6,16,dtype=torch.float64)*.2,torch.randn(6,dtype=torch.float64)]
    x=torch.randn(2,12,dtype=torch.float64)
    _,j=module.value_jacobian(weights,x)
    expected=torch.stack([torch.autograd.functional.jacobian(lambda z:module.value_jacobian(weights,z[None])[0][0],row) for row in x])
    torch.testing.assert_close(j,expected,rtol=1e-10,atol=1e-10)


def test_oracle_inversion_is_bounded_nonincreasing_and_does_not_change_weights():
    torch.manual_seed(3)
    weights=[torch.randn(16,12,dtype=torch.float64)*.2,torch.randn(16,dtype=torch.float64),torch.randn(6,16,dtype=torch.float64)*.2,torch.randn(6,dtype=torch.float64)]
    before=[v.clone() for v in weights];z=torch.randn(3,12,dtype=torch.float64);gain=torch.tensor([.6,1.,1.4],dtype=torch.float64)
    target=module.value_jacobian(weights,z+.1)[0]*gain[:,None]
    result=module.invert_head(weights,z,gain,target,steps=12,radius=.2)
    assert (result['final_error']<=result['initial_error']).all() and (result['shift_norm']<=2.4).all()
    assert (result['final_error']<1e-6).all()
    assert all(torch.equal(a,b) for a,b in zip(weights,before))
