"""Bounded learned-readout gate; oracle data never represents deployed scores."""
import argparse,json,sys,time
from pathlib import Path
import torch,yaml
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.serial_completion import CompletionRelations
from lip.jepa.predictor import SafeAttention
from lip.geometry.so3 import exp,log,update,angle
from lip.engine.jepa_checkpoint import sha


class Readout(nn.Module):
    def __init__(self):
        super().__init__();self.geometry_readout=CompletionRelations()
        self.query=nn.Parameter(torch.randn(1,1,256)*.02)
        self.object_attn=SafeAttention();self.object_norm=nn.LayerNorm(256)
        self.head=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,6))
        nn.init.normal_(self.head[-1].weight,std=1e-4);nn.init.zeros_(self.head[-1].bias)
    def forward(self,packet,base):
        tokens,valid,_,scale=self.geometry_readout(packet,base)
        q=self.query.expand(len(base),-1,-1)
        q=q+self.object_attn(self.object_norm(q),tokens,valid)
        return self.head(F.layer_norm(q[:,0],(256,))).float()*scale[:,None]*valid.any(-1)[:,None]


def batch(records,ids,device='cuda'):
    selected=[records[i] for i in ids]
    pack={k:torch.cat([x['oracle'][k] for x in selected]).to(device) for k in selected[0]['oracle']}
    truth=torch.stack([x['truth'] for x in selected]).to(device)
    d=truth.new_tensor([x['diameter'] for x in selected]);return pack,truth,d


def perturb(packet,truth,d,delta):
    base=update(truth,delta[:,:3],delta[:,3:],d)
    packet=dict(packet,camera=packet['camera']-delta[:,3:,None,None])
    target=torch.cat((log(truth[:,:3,:3]@base[:,:3,:3].transpose(-1,-2)),-delta[:,3:]),-1)
    return packet,base,target


@torch.no_grad()
def evaluate(model,records):
    results={}
    for kind in ('zero','rotation','translation'):
        errors=[];gains=[];centers=[]
        for start in range(0,len(records),8):
            pack,truth,d=batch(records,list(range(start,min(start+8,len(records)))))
            for axis in range(3):
                for sign in (-1,1):
                    delta=truth.new_zeros(len(truth),6)
                    if kind!='zero':delta[:,axis+(3 if kind=='translation' else 0)]=sign*(.05 if kind=='translation' else torch.pi/18)
                    current,base,target=perturb(pack,truth,d,delta)
                    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(current,base)
                    pose=update(base,pred[:,:3],pred[:,3:],d)
                    errors.extend((angle(pose[:,:3,:3]@truth[:,:3,:3].transpose(-1,-2))*180/torch.pi).tolist())
                    centers.extend(((pose[:,:3,3]-truth[:,:3,3]).norm(dim=-1)/d).tolist())
                    if kind!='zero':gains.extend(((pred*target).sum(-1)/target.square().sum(-1)).tolist())
        results[kind]=dict(rotation_deg=sum(errors)/len(errors),center_d=sum(centers)/len(centers),
            signed_gain=sum(gains)/len(gains) if gains else None,cases=len(errors))
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    c=yaml.safe_load(Path(a.config).read_text());records=[];identities={}
    for f in sorted(Path(a.cache).glob('rank*/packets.pt')):
        identities[str(f)]=sha(f);records+=torch.load(f,weights_only=False)
    train=[x for x in records if x['split']=='train'];test=[x for x in records if x['split']=='holdout']
    assert train and test and not ({x['physical'] for x in train}&{x['physical'] for x in test})
    # Keep cache on GPU: this stage trains the small readout, not encoders.
    pack,truth,d=batch(train,list(range(len(train))))
    model=Readout().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();steps=c['serial_completion']['oracle_steps']
    with (out/'training.jsonl').open('w') as log_file:
        for step in range(steps):
            ids=torch.randint(len(train),(32,),device='cuda');p={k:v[ids] for k,v in pack.items()};t=truth[ids];ds=d[ids]
            delta=torch.randn(32,6,device='cuda')*t.new_tensor([.15,.15,.15,.035,.035,.035])
            delta[::4]=0
            p,base,target=perturb(p,t,ds,delta)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(p,base)
            loss=F.smooth_l1_loss(pred[:,:3]/.1745,target[:,:3]/.1745,beta=.1)+F.smooth_l1_loss(pred[:,3:]/.05,target[:,3:]/.05,beta=.1)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if (step+1)%25==0:
                row=dict(step=step+1,loss=float(loss),seconds=time.monotonic()-started);log_file.write(json.dumps(row)+'\n');log_file.flush();print(json.dumps(row),flush=True)
    result=evaluate(model,test);plan=c['serial_completion']
    passed=(result['rotation']['rotation_deg']<=plan['oracle_max_rotation_deg'] and result['rotation']['signed_gain']>=plan['oracle_min_gain'] and
        result['zero']['rotation_deg']<=plan['oracle_zero_rotation_deg'] and result['zero']['center_d']<=plan['oracle_zero_translation_d'])
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=steps,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'readout.pt')
    receipt=dict(completed=True,passed=passed,metrics=result,train_records=len(train),holdout_records=len(test),
        train_physical=sorted({x['physical'] for x in train}),holdout_physical=sorted({x['physical'] for x in test}),
        cache_sha256=identities,checkpoint_sha256=sha(out/'readout.pt'),config_sha256=sha(a.config),steps=steps,
        source='ideal CAD XYZ/depth/features, train split only; NOT native pose performance',seconds=time.monotonic()-started)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
