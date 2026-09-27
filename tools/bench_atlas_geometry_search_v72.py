import json,time,statistics,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.cad_atlas_decoder import geometric_neighbors

torch.set_num_threads(2);torch.manual_seed(72)
xyz=torch.rand(8,8192,3,device='cuda')-.5
prior=xyz[:,:2048].clone()+1e-4*torch.randn(8,2048,3,device='cuda')
valid=torch.ones(8,8192,device='cuda',dtype=torch.bool);valid[:,::7]=False

def direct():
 return torch.cdist(prior,xyz,compute_mode='donot_use_mm_for_euclid_dist').masked_fill(~valid[:,None],float('inf')).topk(8,largest=False).indices

def fast():return geometric_neighbors(prior,xyz,valid)

results={};outputs={}
for name,fn in [('direct',direct),('fast',fast)]:
 fn();torch.cuda.synchronize();times=[]
 for _ in range(3):
  start=time.perf_counter();value=fn();torch.cuda.synchronize();times.append(time.perf_counter()-start)
 results[name]=dict(seconds=times,median_seconds=statistics.median(times));outputs[name]=value
batch=torch.arange(8,device='cuda')[:,None,None]
x=xyz[batch,outputs['direct']];y=xyz[batch,outputs['fast']]
results['index_agreement']=float((outputs['direct']==outputs['fast']).float().mean())
results['max_coordinate_difference']=float((x-y).norm(dim=-1).max())
results['max_neighbor_distance_difference']=float(((prior[:,:,None]-x).norm(dim=-1)-(prior[:,:,None]-y).norm(dim=-1)).abs().max())
assert results['max_neighbor_distance_difference']<1e-6
print(json.dumps(results,indent=2))
