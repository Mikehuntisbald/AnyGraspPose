"""Exercise long rank-0 validation waits without an outstanding NCCL collective."""
from datetime import timedelta
import json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
dist.init_process_group('nccl',timeout=timedelta(seconds=10))
control=dist.new_group(backend='gloo',timeout=timedelta(seconds=60))
x=torch.ones(1,device='cuda');dist.all_reduce(x);torch.cuda.synchronize()
dist.barrier(group=control)
if dist.get_rank()==0:time.sleep(12)
value=[.5 if dist.get_rank()==0 else None];dist.broadcast_object_list(value,0,group=control)
assert value[0]==.5
dist.barrier(group=control)
x.fill_(1);dist.all_reduce(x);assert x.item()==dist.get_world_size()
if dist.get_rank()==0:
 p=Path('runs/full_train_after_upload_20260910/validation_wait_test.json')
 p.write_text(json.dumps(dict(passed=True,world_size=dist.get_world_size(),nccl_timeout_seconds=10,simulated_validation_seconds=12,cpu_control_backend='gloo',post_validation_nccl_all_reduce=float(x.item())),indent=2))
 print(p.read_text(),flush=True)
dist.destroy_process_group(control);dist.destroy_process_group()
