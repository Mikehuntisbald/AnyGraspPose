"""Matched local-refinement training, fixed caches, strict stage resume."""
import argparse,copy,json,sys,hashlib
from pathlib import Path
import torch,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.local_flow import LocalFlowHead
from lip.unified.flow_reconstruction import FlowReconstruction
from lip.engine.jepa_checkpoint import sha
from collect_local_flow_v60 import SOURCE,DIGEST


def exact(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stop-at',type=int,default=1000);p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert 0<a.stop_at<=1000
    torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    data={};physical={};hashes={}
    for split in ('train','holdout'):
        arrays={k:[] for k in ('features','labels','coarse','regions','groups')};records=[];physical[split]=set()
        for rank in range(8):
            path=a.root/'cache'/split/f'rank{rank}'
            receipt=json.loads((path/'receipt.json').read_text())
            assert receipt['completed'] and receipt['source_sha256']==DIGEST and receipt['split']==split
            assert sha(path/'features.pt')==receipt['features_sha256']
            hashes[f'{split}/{rank}']=receipt['features_sha256'];physical[split].update(receipt['physical_sequences'])
            block=torch.load(path/'features.pt',weights_only=True)
            block['groups']+=len(records);records.extend(block['records'])
            for k in arrays:arrays[k].append(block[k])
        data[split]={k:torch.cat(v).cuda() for k,v in arrays.items()};data[split]['records']=records
    assert not physical['train']&physical['holdout']
    assert sha(SOURCE)==DIGEST
    parent=torch.load(SOURCE,map_location='cpu',weights_only=False)
    endpoint=FlowReconstruction().endpoint
    endpoint.load_state_dict({k.removeprefix('flow_reconstruction.endpoint.'):v for k,v in parent['model'].items() if k.startswith('flow_reconstruction.endpoint.')})
    heads={'extra':LocalFlowHead(True)};heads['extra'].initialize(endpoint)
    heads['control']=copy.deepcopy(heads['extra']);heads['control'].use_extra=False
    assert exact(heads['extra'].state_dict(),heads['control'].state_dict())
    heads={k:m.cuda() for k,m in heads.items()};opts={k:torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.01) for k,m in heads.items()}
    contract=dict(cache_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        source_sha256=DIGEST,trainer_sha256=sha(__file__),module_sha256=sha(Path(__file__).resolve().parents[1]/'src/lip/unified/local_flow.py'),
        steps=1000,lr=.001,seed=42,batch=1024,torch=torch.__version__,cuda=torch.version.cuda)
    out=a.root/'training';out.mkdir(exist_ok=a.resume);start=0
    if a.resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert state['contract']==contract
        for k,m in heads.items():
            m.load_state_dict(state['heads'][k]);opts[k].load_state_dict(state['optimizers'][k])
            assert exact(m.state_dict(),state['heads'][k]) and exact(opts[k].state_dict(),state['optimizers'][k])
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);start=state['step']
        assert exact(torch.get_rng_state(),state['rng']) and exact(torch.cuda.get_rng_state_all(),state['cuda_rng'])
        (out/f'resume{start}.json').write_text(json.dumps(dict(weights_exact=True,optimizer_exact=True,rng_exact=True,step=start)))
    else:(out/'provenance.json').write_text(json.dumps(dict(**contract,physical_train_holdout_disjoint=True,backbone_frozen=True,
        loss='pixel SmoothL1 beta1; equal eligible frame-region sampling, proxy weight0.5; both frozen rounds',
        arms='identical pretrained initialization/architecture/minibatches; control zeros added77 dimensions',
        inference_export_only=True),indent=2))
    train=data['train'];weights=torch.zeros(len(train['features']),device='cuda')
    for i,factor in enumerate((1.,1.,.5)):
        counts=torch.bincount(train['groups'],weights=train['regions'][:,i].float(),minlength=len(train['records']))
        weights+=factor*train['regions'][:,i]/counts[train['groups']].clamp_min(1)
    def evaluate(step):
        d=data['holdout'];result={}
        with torch.no_grad():
            for name,m in heads.items():
                pred=d['coarse']+torch.cat([m(x) for x in d['features'].split(4096)])
                error=(pred-d['labels']).norm(dim=-1)
                rows=[]
                for group,record in enumerate(d['records']):
                    row=dict(record,regions={})
                    for i,region in enumerate(('observed','real','proxy')):
                        mask=(d['groups']==group)&d['regions'][:,i]
                        if mask.any():row['regions'][region]=dict(epe=float(error[mask].mean()),within3=float((error[mask]<=3).float().mean()))
                    rows.append(row)
                result[name]=rows
        (out/f'eval{step}.json').write_text(json.dumps(result)+'\n')
    if not start:evaluate(0)
    with (out/'steps.jsonl').open('a') as log:
        for step in range(start,a.stop_at):
            ids=torch.multinomial(weights,1024,replacement=True);row=dict(step=step+1)
            for name,m in heads.items():
                opts[name].zero_grad(set_to_none=True)
                pred=train['coarse'][ids]+m(train['features'][ids])
                loss=F.smooth_l1_loss(pred,train['labels'][ids],beta=1.)
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opts[name].step();row[name]=float(loss.detach())
            log.write(json.dumps(row)+'\n')
            if (step+1)%250==0:evaluate(step+1)
            if (step+1)%50==0 or step+1==a.stop_at:
                state=dict(step=step+1,contract=contract,heads={k:m.state_dict() for k,m in heads.items()},optimizers={k:o.state_dict() for k,o in opts.items()},rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                torch.save(state,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
    if a.stop_at==1000:
        for name,m in heads.items():
            c=copy.deepcopy(parent['config']);c['flow_reconstruction']['local_flow']=dict(enabled=True,use_extra=name=='extra')
            c['local_flow_stage']=dict(inference_export=True,updates=1000,source_sha256=DIGEST)
            state=dict(parent['model']);state.update({'flow_reconstruction.local_flow_head.'+k:v.cpu() for k,v in m.state_dict().items()})
            assert all(torch.equal(state[k],v) for k,v in parent['model'].items())
            torch.save(dict(model=state,config=c,step=parent['step'],local_flow_updates=1000,inference_only=True,architecture_id=parent['architecture_id'],source_sha256=DIGEST),out/f'{name}.pt')
            (out/f'{name}.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
        (out/'complete.json').write_text(json.dumps(dict(completed=True,updates=1000,source_parameters_exact=True)))


if __name__=='__main__':main()
