import torch
from lip.unified.dense_measurement import DenseMeasurementHead

def test_empty_cad_cannot_leak_and_invalid_pixels_are_rejected():
    torch.manual_seed(42);m=DenseMeasurementHead(12).eval()
    rgb=torch.randn(1,3,224,224);ref=torch.randn_like(rgb);geo=torch.randn(1,9,224,224);cad=torch.randn(1,12);available=torch.zeros(1,dtype=torch.bool);bounds=torch.ones(1,1,224,224,dtype=torch.bool);bounds[:,:,:10]=False
    with torch.no_grad():
        a=m(rgb,ref,geo,cad,available,bounds)
        geo[:,2:]=float('nan');b=m(rgb,ref*float('nan'),geo,cad*float('nan'),available,bounds)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert b.isfinite().all() and (b[~bounds]==-30).all()
