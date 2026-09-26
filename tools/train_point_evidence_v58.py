"""Matched frozen-backbone evidence heads; complete head-stage resume state."""
import argparse,copy,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.point_evidence import PointEvidenceHead
from lip.engine.jepa_checkpoint import sha
from collect_point_evidence_v58 import SOURCE,DIGEST


def metrics(score,label):
    good=np.isfinite(label);p=score[good];y=label[good].astype(bool)
    pos,neg=p[y],p[~y]
    # Rank AUC with average ranks for exact ties, O(N log N).
    order=np.argsort(p,kind='stable');sorted_p=p[order]
    ranks=np.empty(len(p),dtype=float);start=0
    while start<len(p):
        stop=start+1
        while stop<len(p) and sorted_p[stop]==sorted_p[start]:stop+=1
        ranks[order[start:stop]]=(start+1+stop)/2;start=stop
    auc=float((ranks[y].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))) if len(pos) and len(neg) else None
    predicted=p>=.5
    return dict(points=len(p),positive=int(y.sum()),auroc=auc,brier=float(np.mean((p-y)**2)),
        precision_at_half=float(y[predicted].mean()) if predicted.any() else None,
        recall_at_half=float(predicted[y].mean()) if y.any() else None,
        predicted_positive=int(predicted.sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-at',type=int,default=1000);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    data={};receipts={};physical={}
    for split in ('train','holdout'):
        blocks=[];physical[split]=set()
        for rank in range(8):
            d=a.root/'cache'/split/f'rank{rank}';receipt=json.loads((d/'receipt.json').read_text())
            assert receipt['completed'] and receipt['source_sha256']==DIGEST and receipt['split']==split
            assert sha(d/'features.pt')==receipt['features_sha256']
            receipts[f'{split}/{rank}']=receipt['features_sha256'];physical[split].update(receipt['physical_sequences'])
            blocks.append(torch.load(d/'features.pt',weights_only=True))
        data[split]={k:torch.cat([b[k] for b in blocks]).cuda() for k in ('features','labels','legacy_logits')}
    assert not physical['train']&physical['holdout']
    source_hash=hashlib.sha256(json.dumps(receipts,sort_keys=True).encode()).hexdigest()
    contract=dict(torch=torch.__version__,cuda=torch.version.cuda,trainer_sha256=sha(__file__),
        head_sha256=sha(Path(__file__).resolve().parents[1]/'src/lip/unified/point_evidence.py'),
        steps=1000,batch_points=1024,learning_rate=1e-3,weight_decay=.01)
    heads={'observed':PointEvidenceHead(True).cuda()};heads['control']=copy.deepcopy(heads['observed']);heads['control'].use_observation=False
    optimizers={k:torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.01,fused=True) for k,m in heads.items()}
    out=a.root/'training';out.mkdir(exist_ok=a.resume)
    start=0
    if a.resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert state['cache_hash']==source_hash and state['source_sha256']==DIGEST and state['contract']==contract
        for k,m in heads.items():m.load_state_dict(state['heads'][k]);optimizers[k].load_state_dict(state['optimizers'][k])
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);start=state['step']
        assert all(torch.equal(v.cpu(),state['heads'][k][n].cpu()) for k,m in heads.items() for n,v in m.state_dict().items())
        def exact(a,b):
            if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
            if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
            return a==b
        assert all(exact(o.state_dict(),state['optimizers'][k]) for k,o in optimizers.items())
        assert exact(torch.get_rng_state(),state['cpu_rng']) and exact(torch.cuda.get_rng_state_all(),state['cuda_rng'])
        (out/f'resume{start}.json').write_text(json.dumps(dict(step=start,cache_hash=source_hash,heads_exact=True,optimizer_exact=True,rng_exact=True)))
    else:
        assert all(torch.equal(x,y) for x,y in zip(heads['observed'].state_dict().values(),heads['control'].state_dict().values()))
        (out/'provenance.json').write_text(json.dumps(dict(source_sha256=DIGEST,cache_hash=source_hash,steps=1000,
            seed=42,learning_rate=1e-3,batch_points=1024,loss='unweighted proper binary cross entropy, separately normalized known labels',
            arms='same architecture/initialization/batches; control zeros139:272 observed evidence',
            backbone_frozen=True,physical_train_holdout_disjoint=True,head_only_stage=True,
            export_inference_only=True,original_reconstruction_targets_unchanged=True),indent=2))
    def evaluate(step):
        x,y=data['holdout']['features'],data['holdout']['labels'];truth=y.cpu().numpy();result={}
        with torch.no_grad():
            for name,m in heads.items():
                score=torch.cat([m(v).sigmoid() for v in x.split(4096)]).cpu().numpy()
                result[name]={label:metrics(score[:,i],truth[:,i]) for i,label in enumerate(('visibility','quality'))}
            old=data['holdout']['legacy_logits'].sigmoid().cpu().numpy()
            result['legacy']={label:metrics(old,truth[:,i]) for i,label in enumerate(('visibility','quality'))}
        (out/f'eval{step}.json').write_text(json.dumps(result,indent=2)+'\n')
    if start==0:evaluate(0)
    x,y=data['train']['features'],data['train']['labels'];begun=time.monotonic()
    with (out/'steps.jsonl').open('a') as log:
        for step in range(start,a.stop_at):
            ids=torch.randint(len(x),(1024,),device='cuda');known=torch.isfinite(y[ids]);target=y[ids].nan_to_num()
            row=dict(step=step+1)
            for name,m in heads.items():
                optimizers[name].zero_grad(set_to_none=True)
                error=F.binary_cross_entropy_with_logits(m(x[ids]),target,reduction='none')
                loss=((error*known).sum(0)/known.sum(0).clamp_min(1)).sum()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite evidence loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizers[name].step()
                row[name]=float(loss.detach())
            log.write(json.dumps(row)+'\n');log.flush()
            if (step+1)%100==0:evaluate(step+1)
            if (step+1)%50==0 or step+1==a.stop_at:
                state=dict(step=step+1,source_sha256=DIGEST,cache_hash=source_hash,contract=contract,heads={k:m.state_dict() for k,m in heads.items()},
                    optimizers={k:o.state_dict() for k,o in optimizers.items()},cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                torch.save(state,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
    if a.stop_at==1000:
        assert sha(SOURCE)==DIGEST
        parent=torch.load(SOURCE,map_location='cpu',weights_only=False)
        for name,m in heads.items():
            config=copy.deepcopy(parent['config']);config['flow_reconstruction']['point_evidence']=dict(enabled=True,use_observation=name=='observed')
            config['point_evidence_stage']=dict(source_sha256=DIGEST,updates=1000,backbone_frozen=True,inference_export=True)
            state=dict(parent['model']);state.update({'flow_reconstruction.point_evidence_head.'+k:v.cpu() for k,v in m.state_dict().items()})
            assert all(torch.equal(v,parent['model'][k]) for k,v in state.items() if k in parent['model'])
            torch.save(dict(model=state,config=config,step=parent['step'],evidence_updates=1000,inference_only=True,
                architecture_id=parent['architecture_id'],model_version='point-evidence-v58-'+name,source_sha256=DIGEST),out/f'{name}.pt')
            (out/f'{name}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        (out/'complete.json').write_text(json.dumps(dict(completed=True,updates=1000,source_weights_exact=True,seconds=time.monotonic()-begun),indent=2))


if __name__=='__main__':main()
