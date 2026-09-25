"""Equal-budget learned readout of ACTUAL frozen JEPA outputs, never oracle XYZ."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from train_serial_oracle import Readout
from lip.geometry.so3 import update,angle,log
from lip.unified.serial_objective import delta_target
from lip.engine.jepa_checkpoint import sha

class CachedReadout(Readout):
    def __init__(self,source,gate='unit'):
        super().__init__();self.source=source;self.gate=gate
        if source=='patch':
            with torch.random.fork_rng():
                torch.manual_seed(43)
                self.geometry_readout.appearance=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128),nn.GELU())
    def forward(self,t):
        g=self.geometry_readout
        tokens=g.combine(torch.cat((g.appearance(t['feature']),g.relation(t['pooled'])),-1))
        tokens=g.norm(torch.cat((tokens,g.moments(t['moments'])[:,None]),1))
        q=self.query.expand(len(tokens),-1,-1);q=q+self.object_attn(self.object_norm(q),tokens,t['token_valid'])
        # Same zero-evidence gate in all arms, without the shape-dependent RMS multiplier.
        gate=(((t['scale']-1e-5).clamp_min(0)*100).tanh() if self.gate=='unit' else t['scale'])*t['token_valid'].any(-1)
        return self.head(F.layer_norm(q[:,0],(256,))).float()*gate[:,None]

def stack(rows,source):
    keys=('pooled','moments','token_valid','scale','base','truth','diameter','original_delta')
    result={k:torch.cat([r[k] for r in rows]).cuda() for k in keys}
    result['feature']=torch.cat([r['feature_'+source] for r in rows]).cuda()
    return result

@torch.no_grad()
def evaluate(model,rows,source):
    values=[]
    for start in range(0,len(rows),32):
        part=rows[start:start+32];t=stack(part,source)
        with torch.autocast('cuda',dtype=torch.bfloat16):delta=model(t)
        pose=update(t['base'],delta[:,:3],delta[:,3:],t['diameter']);old=update(t['base'],t['original_delta'][:,:3],t['original_delta'][:,3:],t['diameter'])
        target=delta_target(t['base'],t['truth'],t['diameter'])
        rotation=angle(pose[:,:3,:3]@t['truth'][:,:3,:3].transpose(1,2))*180/torch.pi
        before=angle(t['base'][:,:3,:3]@t['truth'][:,:3,:3].transpose(1,2))*180/torch.pi
        original=angle(old[:,:3,:3]@t['truth'][:,:3,:3].transpose(1,2))*180/torch.pi
        center=(pose[:,:3,3]-t['truth'][:,:3,3]).norm(dim=-1)/t['diameter']
        gain=(delta[:,:3]*target[:,:3]).sum(-1)/target[:,:3].square().sum(-1).clamp_min(1e-9)
        for i,r in enumerate(part):values.append(dict(case=r['case'],symmetric=r['symmetric'],heavy=r['heavy'],physical=r['physical'],rotation=float(rotation[i]),before=float(before[i]),original=float(original[i]),center_d=float(center[i]),gain=float(gain[i])))
    output={}
    for name,selected in [('rotation',[r for r in values if r['case'] in ('positive','negative')]),('nonsym_rotation',[r for r in values if r['case'] in ('positive','negative') and not r['symmetric']]),('nonsym_augmented_rotation',[r for r in values if r['case'] in ('positive','negative') and not r['symmetric'] and r['heavy']]),('zero',[r for r in values if r['case']=='zero']),('mixed',[r for r in values if r['case']=='mixed'])]:
        if selected:output[name]=dict(cases=len(selected),physical_sequences=len({r['physical'] for r in selected}),**{key:sum(r[key] for r in selected)/len(selected) for key in ('rotation','before','original','center_d','gain')})
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',choices=('observed_visible','decoded','patch'),required=True);p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);rows=[];hashes={}
    for f in sorted(a.cache.glob('rank*/records.pt')):rows+=torch.load(f,weights_only=False);hashes[str(f)]=sha(f)
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='heldout'];assert not {r['physical'] for r in train}&{r['physical'] for r in dev}
    assert len(train)%4==0 and all([r['case'] for r in train[i:i+4]]==['zero','positive','negative','mixed'] for i in range(0,len(train),4))
    model=CachedReadout(a.source).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);resident=stack(train,a.source)
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();generator=torch.Generator().manual_seed(42);metrics=[dict(step=0,heldout=evaluate(model,dev,a.source))]
    for step in range(a.steps):
        group=torch.randint(len(train)//4,(8,),generator=generator).cuda();ids=(4*group[:,None]+torch.arange(4,device='cuda')[None]).flatten();t={k:v[ids] for k,v in resident.items()}
        target=delta_target(t['base'],t['truth'],t['diameter']);scale=target.new_tensor([.174533]*3+[.05]*3)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(t)
        loss=F.smooth_l1_loss(pred/scale,target/scale,beta=.1)
        pairs=pred.reshape(8,4,6);targets=target.reshape(8,4,6)
        loss=loss+.1*F.smooth_l1_loss((pairs[:,1]-pairs[:,2])/scale,(targets[:,1]-targets[:,2])/scale,beta=.1)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (step+1)%250==0:
            m=dict(step=step+1,training=evaluate(model,train,a.source),heldout=evaluate(model,dev,a.source),seconds=time.monotonic()-start);metrics.append(m);print(json.dumps(m),flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),source=a.source,step=a.steps,sampler_rng=generator.get_state()),a.out/'readout.pt')
    receipt=dict(completed=True,source=a.source,steps=a.steps,train_records=len(train),heldout_records=len(dev),cache_sha256=hashes,teacher_geometry_input=False,production_changed=False,metrics=metrics,checkpoint_sha256=sha(a.out/'readout.pt'),scope='Frozen actual-prediction readout probe; physical-sequence-disjoint training-data holdout; not native pose accuracy')
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
