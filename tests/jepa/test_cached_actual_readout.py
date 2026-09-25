import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from train_serial_oracle import Readout
from fit_actual_readout_v27 import CachedReadout


def test_cached_sufficient_statistics_equal_full_geometry_path_and_ignore_labels():
    torch.manual_seed(42);full=Readout().eval();cached=CachedReadout('observed_visible',gate='legacy').eval();cached.load_state_dict(full.state_dict())
    packet=dict(feature=torch.randn(1,256,768),xyz=torch.randn(1,3,224,224)*.1,camera=torch.randn(1,3,224,224)*.1,weight=torch.rand(1,1,224,224),measured_weight=torch.rand(1,1,224,224))
    base=torch.eye(4)[None];values={}
    def capture(name):
        def hook(module,args):values[name]=args[0].detach()
        return hook
    hooks=[full.geometry_readout.relation.register_forward_pre_hook(capture('pooled')),full.geometry_readout.moments.register_forward_pre_hook(capture('moments'))]
    with torch.no_grad():
        _,mask,_,scale=full.geometry_readout(packet,base);expected=full(packet,base)
        t=dict(values,feature=packet['feature'],token_valid=mask,scale=scale,truth=base.clone())
        actual=cached(t);t['truth']*=100;again=cached(t)
    for h in hooks:h.remove()
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(again,actual,rtol=0,atol=0)
