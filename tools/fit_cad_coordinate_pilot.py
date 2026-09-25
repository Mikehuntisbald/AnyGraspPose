"""Same frozen data/order/seed for free DPT versus explicit CAD coordinate reading."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fit_serial_geometry_decoder import stack,loss_and_metrics
from lip.unified.dpt_surface import DPTSurfaceHead
from lip.unified.cad_coordinate_decoder import CADCoordinateDecoder,dense_anchor_loss
from lip.engine.jepa_checkpoint import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--arm',choices=['dpt','cad'],required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);rows=[]
    for f in sorted(Path(a.cache).glob('rank*/decoder_cache.pt')):rows+=torch.load(f,weights_only=False)
    physical=lambda r:'/'.join(r['stream'].split('|')[0].split('/')[:2])
    held=lambda r:int(hashlib.sha256(physical(r).encode()).hexdigest()[:8],16)%4==0
    train=[r for r in rows if not held(r)];dev=[r for r in rows if held(r)]
    assert train and dev and not set(map(physical,train))&set(map(physical,dev))
    model=(CADCoordinateDecoder(rows[0]['cad_features'].shape[-1]) if a.arm=='cad' else DPTSurfaceHead()).cuda()
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    dpt=model.dpt if a.arm=='cad' else model
    dpt.load_state_dict({k.removeprefix('surface_head.'):v for k,v in saved['model'].items() if k.startswith('surface_head.')});del saved
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def forward(t):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if a.arm=='cad':return model(t['levels'],t['valid'],t['cad_features'],t['cad_geometry'],t['cad_available'])
            return model(t['levels'],t['valid']),None
    def evaluate(part):
        stats=[]
        with torch.no_grad():
            for start in range(0,len(part),8):
                t=stack(part[start:start+8]);pred,_=forward(t);_,values=loss_and_metrics(pred,t);stats.append((len(t['valid']),values))
        return {k:sum(n*v[k] for n,v in stats)/len(part) for k in stats[0][1]}
    records=[dict(step=0,train=evaluate(train),heldout=evaluate(dev))]
    generator=torch.Generator().manual_seed(42)
    for step in range(a.steps):
        ids=torch.randint(len(train),(8,),generator=generator).tolist();t=stack([train[i] for i in ids]);pred,logits=forward(t)
        loss,_=loss_and_metrics(pred,t)
        if logits is not None:
            mask=t['real_mask']|t['proxy_mask']|t['observed_mask']
            xyz=torch.where(t['observed_mask'],t['observed_xyz'],t['xyz'])
            loss=loss+.1*dense_anchor_loss(logits,xyz,mask,t['cad_geometry'][:,:,:3],t['cad_available'])
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (step+1)%100==0:
            record=dict(step=step+1,train=evaluate(train),heldout=evaluate(dev),seconds=time.monotonic()-started);records.append(record);print(json.dumps(record),flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),steps=a.steps,arm=a.arm),out/'decoder_diagnostic.pt')
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,arm=a.arm,production_parameters_changed=False,checkpoint_sha256=sha(a.checkpoint),
        train_records=len(train),heldout_records=len(dev),train_physical=len(set(map(physical,train))),heldout_physical=len(set(map(physical,dev))),
        training_split_only=True,lr=3e-4,steps=a.steps,measurements=records,scope='Frozen JEPA decoder pilot; no pose or native claim; CAD arm adds explicit correspondence supervision',seconds=time.monotonic()-started),indent=2)+'\n')

if __name__=='__main__':main()
