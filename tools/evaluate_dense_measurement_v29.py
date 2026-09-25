"""Posthoc selector audit on the exact downstream whole-patch bounds domain.

Original fitting curves used pixel bounds. This separate receipt corrects the
comparison domain without changing or overwriting those archived curves.
"""
import argparse,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from fit_dense_measurement_v29 import stack,predict,DenseMeasurementHead,sha

@torch.no_grad()
def evaluate(model,rows):
    result={}
    for name,part in [('all',rows),('augmented',[r for r in rows if r['heavy']]),('natural',[r for r in rows if not r['heavy']])]:
        counts={kind:dict(tp=0,fp=0,fn=0) for kind in ('original','dense')}
        for start in range(0,len(part),8):
            t=stack(part[start:start+8]);p=predict(model,t).sigmoid()
            bounds=(F.avg_pool2d(t['bounds'].float(),14,14)>=.999).repeat_interleave(14,-2).repeat_interleave(14,-1)
            valid=bounds&t['depth_valid'];target=t['target']
            for kind,prob in [('original',t['original_probability']),('dense',p)]:
                take=(prob>=.7)&valid;v=counts[kind]
                v['tp']+=int((take&target).sum());v['fp']+=int((take&~target).sum());v['fn']+=int((~take&target&valid).sum())
        result[name]={k:dict(**v,precision=v['tp']/max(v['tp']+v['fp'],1),recall=v['tp']/max(v['tp']+v['fn'],1)) for k,v in counts.items()}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    rows=[];hashes={}
    for f in sorted((a.root/'cache').glob('rank*/records.pt')):rows+=torch.load(f,weights_only=False);hashes[str(f)]=sha(f)
    path=a.root/'fit/measurement.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
    model=DenseMeasurementHead(ck['cad_dim']).cuda().eval();model.load_state_dict(ck['model'])
    result=dict(domain='Exact pack_completion domain: full valid 14x14 patch AND finite positive sensor depth; threshold 0.7 unchanged',
        source_checkpoint_sha256=sha(path),cache_sha256=hashes,production_changed=False,pose_evaluated=False,
        metrics={split:evaluate(model,[r for r in rows if r['split']==split]) for split in ('train','heldout')})
    with (a.root/'strict_metrics.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['metrics']['heldout']),flush=True)

if __name__=='__main__':main()
