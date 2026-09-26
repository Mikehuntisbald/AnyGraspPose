"""Verify paired full model/Adam/scheduler/RNG identity, not just file names."""
import argparse,json
from pathlib import Path
import numpy as np,torch

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    checkpoints={arm:torch.load(a.root/arm/'runs/seed42/stage_start.pt',map_location='cpu',weights_only=False) for arm in ('prefix','full')}
    x,y=checkpoints.values();proof={key:equal(x[key],y[key]) for key in ('model','optimizer','scheduler','rng','step','sampler_position')}
    assert all(proof.values()),proof
    xc,yc=x['config'],y['config'];xc['paths']['output']=yc['paths']['output'];xc['training_window']['enabled']=yc['training_window']['enabled'];xc['training_window']['mode']=yc['training_window']['mode']
    assert xc==yc,'Undeclared paired config difference'
    with (a.root/'paired_start_identity.json').open('x') as f:json.dump(dict(proof,config_only_differs_output_and_window_mode=True),f,indent=2)
    print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
