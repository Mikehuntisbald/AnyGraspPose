"""Pair GPU features, predictions and gradients at identical actor/RNG states."""
import json
import os
from pathlib import Path
import subprocess
import time
import torch
import yaml
from lip.data.clips import ClipDataset
from lip.engine.checkpoint import rng_state,restore_rng
from lip.engine.runtime import batch_step
from lip.geometry.renderer import Renderer
from lip.integrations.frozen_fp import FrozenFoundationPose
from lip.models.basin import load_frozen_basin
from lip.models.tracker import Tracker

def main():
    J=Path('runs/basin_speed_v1')
    while not (J/'baseline_repeatability.json').exists():time.sleep(2)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(2)
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(42)
    c=yaml.safe_load((J/'optimized.yaml').read_text());root='cache/raw_full_20260910'
    model=Tracker(False).cuda();ck=torch.load(J/'resume_21500.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model']);model.train()
    critic=load_frozen_basin(c['basin_checkpoint'],'cuda',c['basin_checkpoint_sha256'])
    fp=FrozenFoundationPose(c['foundationpose_root'],root,torch.device('cuda',0),c['foundationpose_refiner_sha256'])
    renderer=Renderer('cuda')
    ds=ClipDataset(root,'cache/dexycb_s0',steps=8,batch=32,world=8,start=21505,decode_threads=4)
    items=ds.__getitems__(list(range(8)));rows=[]
    for mode in ('noisy_gt','lip_only','lip_fp'):
        features=[];outputs=[]
        def remember(u,inp,out):
            features.append({k:v.detach().clone() for k,v in inp.items()})
            outputs.append({k:v.detach().clone() for k,v in out.items()})
        model.zero_grad(set_to_none=True);state=rng_state()
        a,_=batch_step(model,items,renderer,dict(c,preload_rollout_observations=False),4,True,True,
                      history_mode=mode,fp_transition=fp,basin_critic=critic,basin_weight=.01,observer=remember)
        gradients={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        checks=[]
        def compare(u,inp,out):
            assert all(torch.equal(v,features[u][k]) for k,v in inp.items()), f'{mode}: model inputs differ'
            checks.append(max((v.float()-outputs[u][k].float()).abs().max().item() for k,v in out.items()))
        model.zero_grad(set_to_none=True);restore_rng(state)
        b,_=batch_step(model,items,renderer,dict(c,preload_rollout_observations=True),4,True,True,
                      history_mode=mode,fp_transition=fp,basin_critic=critic,basin_weight=.01,observer=compare)
        diff=sum((p.grad.float()-gradients[n]).square().sum().item() for n,p in model.named_parameters() if n in gradients)
        norm=sum(v.square().sum().item() for v in gradients.values())
        relative=(diff/max(norm,1e-30))**.5
        assert abs(a['loss']-b['loss'])<1e-6 and max(checks)<1e-6 and relative<1e-5
        assert all(p.grad is None for p in critic.parameters()) and all(p.grad is None for p in fp.refiner.model.parameters())
        row=dict(mode=mode,clips=8,rollout=4,inputs_bitwise_equal=True,max_output_difference=max(checks),
                 loss_difference=abs(a['loss']-b['loss']),gradient_relative_l2=relative,frozen_modules_have_no_gradients=True)
        rows.append(row);print(json.dumps(row),flush=True)
    (J/'gpu_equivalence.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))

if __name__=='__main__':main()
