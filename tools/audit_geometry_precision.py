"""Counterexample: identical CAD/GT projection must yield zero flow under AMP."""
import hashlib
import json
import torch
import lip.unified.cad_transport as module

torch.set_num_threads(2)
grid=module.pixel_grid(1,224,224,'cuda')
k=torch.tensor([[[500.,0,111.5],[0,500.,111.5],[0,0,1.]]],device='cuda')
xyz=torch.cat(((grid-111.5)/500,torch.zeros_like(grid[:,:1])),1)
base=torch.eye(4,device='cuda')[None];base[:,2,3]=1
geometry=torch.zeros(1,9,224,224,device='cuda');geometry[:,3]=1;geometry[:,4:7]=xyz
valid=torch.ones(1,256,device='cuda',dtype=torch.bool);diameter=torch.ones(1,device='cuda')
fp=module.transport_targets(xyz,geometry,base,diameter,k,valid)
with torch.autocast('cuda',dtype=torch.bfloat16):
    amp=module.transport_targets(xyz,geometry,base,diameter,k,valid)
print(json.dumps(dict(source_file=module.__file__,source_sha256=hashlib.file_digest(open(module.__file__,'rb'),'sha256').hexdigest(),
    fp32_max_abs_flow_px=float(fp['flow'].abs().max()),amp_max_abs_flow_px=float(amp['flow'].abs().max()),
    mean_amp_difference_px=float((fp['flow']-amp['flow']).norm(dim=1).mean()),
    all_outputs_exact=all(torch.equal(fp[name],amp[name]) for name in fp)),indent=2))
