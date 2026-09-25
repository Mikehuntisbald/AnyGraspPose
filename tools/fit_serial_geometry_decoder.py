"""Frozen-JEPA decoder fitting diagnostic, not production model training."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.dpt_surface import DPTSurfaceHead
from lip.unified.recovery_focus import masked_mean
from lip.engine.jepa_checkpoint import sha


def stack(rows):
    keys=[k for k,v in rows[0].items() if isinstance(v,torch.Tensor)]
    batch={k:torch.cat([r[k] for r in rows]).cuda() for k in keys}
    batch['levels']=[torch.cat([r['levels'][i] for r in rows]).cuda() for i in range(4)]
    batch['diameter']=torch.tensor([r['diameter'] for r in rows],device='cuda')
    return batch


def loss_and_metrics(pred,t):
    geometry=t['real_mask']|t['proxy_mask'];known=torch.isfinite(t['validity'])
    xyz=F.smooth_l1_loss(pred[:,:3],t['xyz'],beta=.05,reduction='none').mean(1,keepdim=True)
    dep=F.smooth_l1_loss(pred[:,3:4],t['depth'],beta=.05,reduction='none')
    visible=F.smooth_l1_loss(pred[:,:3],t['observed_xyz'],beta=.05,reduction='none').mean(1,keepdim=True)
    camera=torch.einsum('bij,bjhw->bihw',t['rotation'],pred[:,:3])+t['translation_d'][:,:,None,None]-t['rays']*(pred[:,3:4]+t['base_depth_d'][:,None,None,None])
    consistency=F.smooth_l1_loss(camera,torch.zeros_like(camera),beta=.05,reduction='none').mean(1,keepdim=True)
    validity=masked_mean(F.binary_cross_entropy_with_logits(pred[:,4:5],t['validity'].nan_to_num(),reduction='none'),known)
    loss=masked_mean(xyz+dep+consistency,geometry)+masked_mean(visible,t['observed_mask'])+.1*validity
    scale=t['diameter'][:,None,None,None]*1000
    with torch.no_grad():
        results={}
        for name,mask in [('real',t['real_mask']),('proxy',t['proxy_mask'])]:
            # masked_mean includes empty lanes; retain eligible pixels for separate audit.
            results[name+'_xyz_mm']=float(masked_mean((pred[:,:3]-t['xyz']).norm(dim=1,keepdim=True)*scale,mask))
            results[name+'_depth_mm']=float(masked_mean((pred[:,3:4]-t['depth']).abs()*scale,mask))
        results['visible_xyz_mm']=float(masked_mean((pred[:,:3]-t['observed_xyz']).norm(dim=1,keepdim=True)*scale,t['observed_mask']))
    return loss,results


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=500);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    rows=[]
    for file in sorted(Path(a.cache).glob('rank*/decoder_cache.pt')):rows+=torch.load(file,weights_only=False)
    # Repeatable physical-sequence split inside the training split.
    held=lambda r:int(hashlib.sha256('/'.join(r['stream'].split('/')[:2]).encode()).hexdigest()[:8],16)%4==0
    train=[r for r in rows if not held(r)];test=[r for r in rows if held(r)]
    assert train and test
    model=DPTSurfaceHead().cuda();saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict({k.removeprefix('surface_head.'):v for k,v in saved['model'].items() if k.startswith('surface_head.')});del saved
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def evaluate(part):
        stats=[]
        with torch.no_grad():
            for start in range(0,len(part),8):
                t=stack(part[start:start+8])
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(t['levels'],t['valid'])
                _,values=loss_and_metrics(pred,t);stats.append((len(t['valid']),values))
        return {k:sum(n*v[k] for n,v in stats)/len(part) for k in stats[0][1]}
    measurements=[dict(step=0,train=evaluate(train),heldout=evaluate(test))]
    resident=stack(train)
    for step in range(a.steps):
        ids=torch.randint(len(train),(8,),device='cuda')
        t={k:([v[ids] for v in val] if k=='levels' else val[ids]) for k,val in resident.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(t['levels'],t['valid'])
        loss,_=loss_and_metrics(pred,t);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (step+1)%100==0:
            row=dict(step=step+1,train=evaluate(train),heldout=evaluate(test),seconds=time.monotonic()-started);measurements.append(row);print(json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),steps=a.steps),out/'decoder_diagnostic.pt')
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,production_parameters_changed=False,source_checkpoint_sha256=sha(a.checkpoint),
        training_records=len(train),heldout_records=len(test),training_split_only=True,scope='Cached frozen JEPA and fixed coarse routing; decoder capacity diagnostic, not native accuracy',
        measurements=measurements,seconds=time.monotonic()-started),indent=2)+'\n')

if __name__=='__main__':main()
