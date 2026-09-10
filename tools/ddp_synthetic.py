"""Distributed engine correctness only; cannot approve real-data preflight."""
import argparse,json,os,time
from pathlib import Path
import torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lip.data.synthetic import synthetic_item
from lip.engine.config import load_config,optimizer_and_scheduler
from lip.engine.runtime import batch_step
from lip.engine.checkpoint import save,resume
from lip.models.tracker import Tracker
from lip.geometry.renderer import Renderer
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke.yaml');p.add_argument('--steps',type=int,default=50);p.add_argument('--out',default='runs/ddp_synthetic');p.add_argument('--resume');a=p.parse_args()
local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);torch.cuda.set_device(local);dist.init_process_group('nccl');c=load_config(a.config)
torch.set_num_threads(1);torch.manual_seed(42+rank);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
model=Tracker(c['pretrained']).cuda();optimizer,scheduler=optimizer_and_scheduler(model,c);audit=dict(mesh_hash='synthetic_box_fixture_v1',split_hash='NOT_DEXYCB');step=0
if a.resume:step=resume(a.resume,model,optimizer,scheduler,audit,rank)['global_step']
model=DDP(model,device_ids=[local],broadcast_buffers=False);renderer=Renderer(torch.device('cuda',local));item=synthetic_item(c['clip_length']);item['seed']+=rank
log=(out/f'rank{rank}.jsonl').open('a')
while step<a.steps:
 optimizer.zero_grad(set_to_none=True);u=4 if step%2 else 1;t=time.perf_counter()
 # Exercise no_sync across accumulation as well as across rollout steps.
 for micro in range(2):
  values,_=batch_step(model,[item],renderer,c,rollout=u,loss_scale=.5,sync_final=micro==1)
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step();step+=1
 row=dict(source='synthetic_correctness_only',rank=rank,step=step,scheduler_step=scheduler.last_epoch,rollout=u,samples_seen=step*2,seconds=time.perf_counter()-t,grad_norm=float(norm),**values)
 log.write(json.dumps(row)+'\n');log.flush()
 if rank==0:print(json.dumps(row),flush=True)
save(out/'last.pt',model,optimizer,scheduler,step,c,audit,step*2,None)
check=torch.tensor([step,scheduler.last_epoch],device='cuda');checks=[torch.zeros_like(check) for _ in range(dist.get_world_size())];dist.all_gather(checks,check)
if rank==0:(out/'receipt.json').write_text(json.dumps(dict(source='synthetic_correctness_only',passed=all(torch.equal(check,x) for x in checks),rank_steps=[x.tolist() for x in checks],resume=bool(a.resume)),indent=2))
dist.destroy_process_group()
