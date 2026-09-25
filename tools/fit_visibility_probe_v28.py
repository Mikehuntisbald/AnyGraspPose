"""Equal-capacity visibility probes; guard against winning by discarding measurements."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.jepa_checkpoint import sha

def stack(rows,source):
    keys=('target','valid','depth_count','visible_depth_count','original_probability')
    return dict({k:torch.cat([r[k] for r in rows]).cuda() for k in keys},feature=torch.cat([r['feature_'+source] for r in rows]).cuda())

@torch.no_grad()
def evaluate(model,rows,source):
    result={}
    for split,part in [('all',rows),('augmented',[r for r in rows if r['heavy']]),('natural',[r for r in rows if not r['heavy']])]:
        sums={name:dict(tp=0.,fp=0.,fn=0.,bce=0.,pixels=0.,selected=0.,valid=0.) for name in ('original','probe')}
        for start in range(0,len(part),64):
            t=stack(part[start:start+64],source)
            with torch.autocast('cuda',dtype=torch.bfloat16):prob=model(t['feature']).squeeze(-1).float().sigmoid()
            for name,p in [('original',t['original_probability']),('probe',prob)]:
                selected=(p>=.7)&t['valid'];v=sums[name]
                v['tp']+=float((t['visible_depth_count']*selected).sum());v['fp']+=float(((t['depth_count']-t['visible_depth_count'])*selected).sum())
                v['fn']+=float((t['visible_depth_count']*~selected*t['valid']).sum())
                v['bce']+=float((F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),t['target'],reduction='none')*t['valid']).sum())
                v['valid']+=int(t['valid'].sum());v['selected']+=int(selected.sum())
        result[split]={}
        for name,v in sums.items():result[split][name]=dict(bce=v['bce']/max(v['valid'],1),measurement_precision=v['tp']/max(v['tp']+v['fp'],1),measurement_recall=v['tp']/max(v['tp']+v['fn'],1),selected_patch_fraction=v['selected']/max(v['valid'],1),true_selected_depth_pixels=v['tp'],false_selected_depth_pixels=v['fp'])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',choices=('raw','fused','patch'),required=True);p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);rows=[];hashes={}
    for f in sorted(a.cache.glob('rank*/records.pt')):rows+=torch.load(f,weights_only=False);hashes[str(f)]=sha(f)
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='heldout'];assert not {r['physical'] for r in train}&{r['physical'] for r in dev}
    model=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,1)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01);t=stack(train,a.source);generator=torch.Generator().manual_seed(42)
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();metrics=[dict(step=0,heldout=evaluate(model,dev,a.source))]
    for step in range(a.steps):
        ids=torch.randint(len(train),(64,),generator=generator).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(t['feature'][ids]).squeeze(-1).float()
        loss=(F.binary_cross_entropy_with_logits(logits,t['target'][ids],reduction='none')*t['valid'][ids]).sum()/t['valid'][ids].sum().clamp_min(1)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if (step+1)%250==0:
            row=dict(step=step+1,training=evaluate(model,train,a.source),heldout=evaluate(model,dev,a.source),seconds=time.monotonic()-start);metrics.append(row);print(json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),source=a.source,step=a.steps),a.out/'visibility.pt')
    receipt=dict(completed=True,source=a.source,steps=a.steps,cache_sha256=hashes,metrics=metrics,checkpoint_sha256=sha(a.out/'visibility.pt'),production_changed=False,scope='Training-sequence holdout calibration only; threshold0.7 unchanged; precision AND recall required, no pose claim')
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
