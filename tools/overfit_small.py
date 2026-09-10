import argparse,json,os,time
from pathlib import Path
import torch
from lip.data.clips import ClipDataset
from lip.engine.config import load_config,check_data_gate,optimizer_and_scheduler
from lip.engine.runtime import batch_step
from lip.engine.checkpoint import save
from lip.models.tracker import Tracker
from lip.geometry.renderer import Renderer
from lip.evaluation.metrics import errors
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--num-clips',type=int,default=32);p.add_argument('--steps',type=int,default=500);p.add_argument('--out',default='runs/overfit32');p.add_argument('--data-root',default=os.getenv('DEX_YCB_DIR'));p.add_argument('--index-root',default='cache/dexycb_s0');p.add_argument('--allow-verified-subset',action='store_true');a=p.parse_args()
if not a.data_root:p.error('DEX_YCB_DIR required')
c=load_config(a.config);audit=check_data_gate(a.index_root,a.allow_verified_subset);torch.set_num_threads(c.get('cpu_threads',2));torch.manual_seed(c['seed'])
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ds=ClipDataset(a.data_root,a.index_root,c['clip_length'],augmentation=False,seed=c['seed'])
# Freeze the same declared object/sequence/camera/bucket sampler used by training.
# Enforce unique clip identities with a bounded traversal of deterministic draws.
import numpy as np
if len(ds.pool)<a.num_clips:raise RuntimeError('Fewer than requested unique real clips')
chosen=[];seen=set()
for draw in range(max(len(ds.pool)*4,a.num_clips*100)):
 sample,_=ds.choose(draw);key=(sample['stream'],sample['start'],sample['stride'])
 if key not in seen:chosen.append(sample);seen.add(key)
 if len(chosen)==a.num_clips:break
if len(chosen)!=a.num_clips:raise RuntimeError('Could not obtain the requested unique fixed clips')
ds.fixed=chosen;items=[ds[i] for i in range(a.num_clips)]
sample_records=[dict(stream=item['stream']['stream_id'],object_id=item['stream']['object_id'],frames=item['frames'].tolist()) for item in items]
(out/'manifest.json').write_text(json.dumps(dict(seed=c['seed'],clips=ds.fixed,source='real_DexYCB',split_hash=audit['split_hash'],noise=False,augmentation=False,sampling='declared mixture; object/sequence/camera balanced; fixed unique clips',sample_records=sample_records),indent=2))
model=Tracker(c['pretrained'],dropout=0.).cuda();renderer=Renderer('cuda');optimizer,scheduler=optimizer_and_scheduler(model,c)
bs=min(4,a.num_clips);log=(out/'loss.jsonl').open('w');history=[]
def measure():
 model.eval();ls=[];preds=[];baseline=[]
 with torch.no_grad():
  for i in range(0,len(items),bs):
   batch=items[i:i+bs];values,outs=batch_step(model,batch,renderer,c,noise=False,backward=False);ls.append(values['loss'])
   for item,pred in zip(batch,outs[0]):
    gt=item['poses'][c['clip_length']];base=item['poses'][c['clip_length']-1]
    preds.append(errors(pred.cpu(),gt,item['mesh']['vertices'],float(item['mesh']['diameter'])))
    baseline.append(errors(base,gt,item['mesh']['vertices'],float(item['mesh']['diameter'])))
 return dict(loss=float(np.mean(ls)),learned={k:float(np.mean([r[k] for r in preds])) for k in preds[0]},zero_motion={k:float(np.mean([r[k] for r in baseline])) for k in baseline[0]})
initial=measure();model.train()
for step in range(a.steps):
 batch=[items[(step*bs+j)%len(items)] for j in range(bs)];optimizer.zero_grad(set_to_none=True)
 values,_=batch_step(model,batch,renderer,c,noise=False);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step()
 values.update(step=step+1);log.write(json.dumps(values)+'\n');log.flush()
 if (step+1)%25==0:print(json.dumps(values),flush=True)
final=measure();passed=final['loss']<initial['loss']*.75 and final['learned']['add_m']<final['zero_motion']['add_m']
report=dict(passed=passed,data_complete=audit['complete'],split_scope=audit['inventory_scope'],source='real_DexYCB',num_clips=a.num_clips,steps=a.steps,initial=initial,final=final,criterion='loss <75% initial and learned ADD below identical zero-motion baseline')
(out/'report.json').write_text(json.dumps(report,indent=2));save(out/'last.pt',model,optimizer,scheduler,a.steps,c,audit,a.steps,None);print(json.dumps(report,indent=2))
if not passed:raise SystemExit(2)
