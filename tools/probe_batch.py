import argparse,gc,json,os,time
from pathlib import Path
import torch,yaml
from lip.data.clips import ClipDataset
from lip.data.synthetic import synthetic_item
from lip.engine.config import load_config,check_data_gate,environment,optimizer_and_scheduler
from lip.engine.runtime import batch_step,sync
from lip.models.tracker import Tracker
from lip.geometry.renderer import Renderer
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',default='configs/resolved_8gpu.yaml');p.add_argument('--data-root',default=os.getenv('DEX_YCB_DIR'));p.add_argument('--index-root',default='cache/dexycb_s0');p.add_argument('--synthetic-memory-only',action='store_true');p.add_argument('--batches',type=int,nargs='+',default=[4,8,16,32]);p.add_argument('--allow-verified-subset',action='store_true');a=p.parse_args()
c=load_config(a.config);torch.set_num_threads(c.get('cpu_threads',2));torch.manual_seed(c['seed']);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
if a.synthetic_memory_only:
 if out.name=='resolved_8gpu.yaml':p.error('Synthetic probe cannot approve resolved_8gpu.yaml; choose a candidate output')
 template=synthetic_item(c['clip_length']);get=lambda n:[template]*n;audit=None
else:
 if not a.data_root:p.error('DEX_YCB_DIR required')
 audit=check_data_gate(a.index_root,a.allow_verified_subset);
 if not audit['complete'] and out.name=='resolved_8gpu.yaml':p.error('Partial data cannot approve full-s0 resolved_8gpu.yaml')
 ds=ClipDataset(a.data_root,a.index_root,c['clip_length'],augmentation=False);get=lambda n:[ds[i] for i in range(n)]
renderer=Renderer('cuda');free,total=torch.cuda.mem_get_info();rows=[]
for b in a.batches:
 model=None;optimizer=None
 try:
  model=Tracker(c['pretrained']).cuda();optimizer,_=optimizer_and_scheduler(model,c);items=get(b)
  for u in [1,4]:
   torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True);sync(torch.device('cuda'));start=time.perf_counter()
   values,_=batch_step(model,items,renderer,c,rollout=u,noise=True)
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();sync(torch.device('cuda'))
   row=dict(batch=b,rollout=u,passed=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),seconds=time.perf_counter()-start,**values)
   row['memory_safe']=row['peak_reserved']<.8*free;rows.append(row);print(json.dumps(row),flush=True)
 except torch.cuda.OutOfMemoryError as e:
  rows.append(dict(batch=b,rollout=u,passed=False,error='CUDA OOM'));print(json.dumps(rows[-1]),flush=True)
 finally:
  del model,optimizer;gc.collect();torch.cuda.empty_cache()
valid=[b for b in a.batches if 256%(8*b)==0 and all(any(r['batch']==b and r['rollout']==u and r.get('passed') and r['memory_safe'] for r in rows) for u in [1,4])]
report=dict(source='synthetic_memory_only' if a.synthetic_memory_only else 'real_DexYCB',data_complete=bool(audit and audit['complete']),environment=environment(),available_bytes_at_start=free,results=rows,eligible_batches=valid)
if valid:
 chosen=max(valid);c.update(batch_size_per_gpu=chosen,grad_accum_steps=256//(8*chosen),probe_source=report['source'],data_geometry_passed=not a.synthetic_memory_only,preflight_approved=False)
 if audit:c.update(split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'])
 out.write_text(yaml.safe_dump(c,sort_keys=False))
out.with_suffix('.probe.json').write_text(json.dumps(report,indent=2))
if not valid:raise SystemExit(2)
