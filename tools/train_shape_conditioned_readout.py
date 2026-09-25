"""Same-parent readout-only comparison on train-split ideal geometry.

All arms use cached STUDENT appearance. This teacher-forcing experiment cannot
establish deployed restoration/pose accuracy; a separate controlled val follows.
"""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from train_serial_oracle import Readout,batch,perturb,evaluate
from lip.unified.serial_completion import CompletionRelations
from lip.unified.reconstruction_only import is_pose_parameter
from lip.engine.jepa_checkpoint import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--conditioning',choices=['legacy','unit_gate','shape_conditioned'],required=True);p.add_argument('--steps',type=int,default=600);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);records=[];cache_hashes={}
    for f in sorted(Path(a.cache).glob('rank*/packets.pt')):
        cache_hashes[str(f)]=sha(f);records+=torch.load(f,weights_only=False)
    for r in records:
        r['oracle']['feature']=r['predicted']['feature']
        for k in ('weight','completed_weight'):r['oracle'][k]=.5*r['oracle'][k]
    train=[r for r in records if r['split']=='train'];dev=[r for r in records if r['split']=='holdout']
    assert train and dev and not {r['physical'] for r in train}&{r['physical'] for r in dev}
    model=Readout().cuda();model.geometry_readout=CompletionRelations(a.conditioning).cuda()
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    states={k:v for k,v in source['model'].items() if is_pose_parameter(k)};del source
    status=model.load_state_dict(states,strict=False)
    assert not status.unexpected_keys and all(k.startswith('geometry_readout.precondition.') for k in status.missing_keys)
    for k,v in states.items():assert torch.equal(model.state_dict()[k].cpu(),v)
    del states
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    metrics=[dict(step=0,heldout=evaluate(model,dev))]
    resident,truth,diameter=batch(train,list(range(len(train))))
    generator=torch.Generator().manual_seed(42)
    with (out/'training.jsonl').open('w') as f:
        for step in range(a.steps):
            ids=torch.randint(len(train),(32,),generator=generator).cuda();packet={k:v[ids] for k,v in resident.items()}
            t=truth[ids];d=diameter[ids]
            noise=torch.randn(32,6,generator=generator).cuda()*t.new_tensor([.15]*3+[.035]*3);noise[::4]=0
            packet,base,target=perturb(packet,t,d,noise)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(packet,base)
            loss=F.smooth_l1_loss(pred[:,:3]/.1745,target[:,:3]/.1745,beta=.1)+F.smooth_l1_loss(pred[:,3:]/.05,target[:,3:]/.05,beta=.1)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if (step+1)%50==0:
                row=dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-started);f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
            if (step+1)%300==0:metrics.append(dict(step=step+1,heldout=evaluate(model,dev)))
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=a.steps,conditioning=a.conditioning,source_sha256=sha(a.checkpoint),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler_rng=generator.get_state()),out/'readout.pt')
    result=dict(completed=True,conditioning=a.conditioning,source_checkpoint_sha256=sha(a.checkpoint),train_records=len(train),heldout_records=len(dev),
        cache_sha256=cache_hashes,metrics=metrics,steps=a.steps,checkpoint_sha256=sha(out/'readout.pt'),optimizer_lr=3e-4,
        shared_starting_parameters_exact=True,teacher_geometry=True,appearance='cached student',production_changed=False,
        scope='Train-split sequence-disjoint readout gate only; not native or predicted-geometry accuracy',seconds=time.monotonic()-started)
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
