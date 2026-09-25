"""Pixel measurement selection pilot; frozen JEPA and no pose training."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.dense_measurement import DenseMeasurementHead
from lip.engine.jepa_checkpoint import sha

def stack(rows):
    return {k:torch.cat([r[k] for r in rows]).cuda() for k in ('rgb','reference','geometry','cad','available','bounds','target','depth_valid','original_probability')}

def predict(model,t):
    with torch.autocast('cuda',dtype=torch.bfloat16):return model(t['rgb'],t['reference'],t['geometry'],t['cad'],t['available'],t['bounds'])

@torch.no_grad()
def evaluate(model,rows):
    result={}
    for name,part in [('all',rows),('augmented',[r for r in rows if r['heavy']]),('natural',[r for r in rows if not r['heavy']])]:
        totals={kind:dict(tp=0.,fp=0.,fn=0.,bce=0.,count=0.) for kind in ('original','dense')}
        for start in range(0,len(part),8):
            t=stack(part[start:start+8]);p=predict(model,t).sigmoid()
            for kind,prob in [('original',t['original_probability']),('dense',p)]:
                valid=t['bounds']&t['depth_valid'];take=(prob>=.7)&valid;target=t['target'];v=totals[kind]
                v['tp']+=int((take&target).sum());v['fp']+=int((take&~target).sum());v['fn']+=int((~take&target&valid).sum())
                v['bce']+=float((F.binary_cross_entropy(prob.clamp(1e-6,1-1e-6),target.float(),reduction='none')*t['bounds']).sum());v['count']+=int(t['bounds'].sum())
        result[name]={kind:dict(precision=v['tp']/max(v['tp']+v['fp'],1),recall=v['tp']/max(v['tp']+v['fn'],1),bce=v['bce']/max(v['count'],1),true_pixels=v['tp'],false_pixels=v['fp']) for kind,v in totals.items()}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=1000);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    rows=[];hashes={}
    for f in sorted(a.cache.glob('rank*/records.pt')):rows+=torch.load(f,weights_only=False);hashes[str(f)]=sha(f)
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='heldout'];assert not {r['physical'] for r in train}&{r['physical'] for r in dev}
    model=DenseMeasurementHead(rows[0]['cad'].shape[-1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01);generator=torch.Generator().manual_seed(42)
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();metrics=[dict(step=0,heldout=evaluate(model,dev))]
    for step in range(a.steps):
        ids=torch.randint(len(train),(8,),generator=generator).tolist();t=stack([train[i] for i in ids]);logits=predict(model,t);target=t['target'].float();mask=t['bounds'].float();prob=logits.sigmoid()
        bce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
        dice=(1-(2*(prob*target*mask).flatten(1).sum(-1)+1)/((prob.square()+target)*mask).flatten(1).sum(-1).add(1)).mean()
        loss=bce+.5*dice;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (step+1)%250==0:
            row=dict(step=step+1,training=evaluate(model,train),heldout=evaluate(model,dev),seconds=time.monotonic()-start);metrics.append(row);print(json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=a.steps,cad_dim=rows[0]['cad'].shape[-1],sampler_rng=generator.get_state()),a.out/'measurement.pt')
    rec=dict(completed=True,steps=a.steps,metrics=metrics,cache_sha256=hashes,checkpoint_sha256=sha(a.out/'measurement.pt'),production_changed=False,
        inputs='Actual RGB-D, estimated-pose CAD RGB/geometry, frozen static Utonia descriptor; no predicted completion or trainable DINO feature',scope='Pixel selection on disjoint training sequences; BCE+0.5 Dice; fixed threshold0.7; no pose improvement claim')
    (a.out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')

if __name__=='__main__':main()
