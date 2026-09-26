import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from score_official_flow_v65 import sample_prediction

def test_resize_flow_units_identity_and_valid_normalization():
    reference=torch.tensor([[[6.,6.],[111.,117.],[215.,213.]]])
    mask=torch.ones(1,280,280)
    p=dict(flow=torch.zeros(1,2,280,280),confidence=mask*.7,mask=mask)
    uv,confidence,mass=sample_prediction(p,reference)
    torch.testing.assert_close(uv,reference);torch.testing.assert_close(confidence,torch.full_like(confidence,.7))
    p['flow'][:,0]=5.;p['flow'][:,1]=-2.
    uv,_,_=sample_prediction(p,reference)
    torch.testing.assert_close(uv,reference+torch.tensor([4.,-1.6]))
    # Fractional valid mass cannot attenuate a constant valid displacement.
    p={k:v*.3 for k,v in p.items()}
    uv,confidence,_=sample_prediction(p,reference)
    torch.testing.assert_close(uv,reference+torch.tensor([4.,-1.6]))
    torch.testing.assert_close(confidence,torch.full_like(confidence,.7))
