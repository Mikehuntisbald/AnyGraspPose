"""Offline statistics; optional torchrun sharding with an explicit merge step."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from lip.geometry.so3 import center_pose,angle
from lip.geometry.renderer import Renderer
from lip.evaluate import visibility
p=argparse.ArgumentParser();p.add_argument('--data-root',default=os.getenv('DEX_YCB_DIR'));p.add_argument('--index',default='cache/dexycb_s0');p.add_argument('--length',type=int,default=8);p.add_argument('--device',default='cuda');p.add_argument('--test-finalized',action='store_true');a=p.parse_args()
if not a.data_root:p.error('DEX_YCB_DIR required')
rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'));local=int(os.getenv('LOCAL_RANK','0'))
if a.device.startswith('cuda'):torch.cuda.set_device(local);a.device=f'cuda:{local}'
torch.set_num_threads(1)
index=Path(a.index);renderer=Renderer(a.device);streams=[json.loads(x) for x in (index/'streams.jsonl').read_text().splitlines()]
audit=json.loads((index/'audit.json').read_text());shards=index/'sampling_shards';shards.mkdir(exist_ok=True)
for split in ['train','val','test']:
 if split=='test' and not a.test_finalized:continue
 pool=[];selected=[s for s in streams if s['split']==split]
 for sid,s in enumerate(selected):
  if sid%world!=rank:continue
  with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
  with np.load(index/s['pose_cache']) as z:frames=z['frames'].copy();poses=center_pose(torch.from_numpy(z['poses']),torch.from_numpy(mesh['center']))
  valid=set(frames.tolist());vis={};k=torch.tensor(s['intrinsics'],device=a.device);mapping={int(f):i for i,f in enumerate(frames)}
  for stride in [1,2,4]:
   for start in frames:
    ts=list(range(int(start)-stride,int(start)+(a.length+3)*stride,stride))
    if not all(t in valid for t in ts):continue
    end=int(start)+(a.length-1)*stride
    if end not in vis:vis[end]=visibility(a.data_root,s,end,poses[mapping[end]].to(a.device),k,mesh,renderer)
    pp=poses[[mapping[t] for t in ts[1:a.length+1]]];base=pp[0]
    moving=bool(((pp[:,:3,3]-base[:3,3]).norm(dim=-1)/float(mesh['diameter'])>.05).any() or
                (angle(pp[:,:3,:3]@base[:3,:3].T)>5*torch.pi/180).any())
    pool.append(dict(stream=sid,start=int(start),stride=stride,moving=moving,occluded=vis[end] is not None and vis[end]<.3,visibility=vis[end]))
  if sid//world%16==0:print(json.dumps(dict(rank=rank,split=split,stream=sid,total_streams=len(selected),clips=len(pool))),flush=True)
 dest=shards/f'{split}_L{a.length}_rank{rank}.json'
 tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(dict(rank=rank,world=world,split=split,split_hash=audit['split_hash'],clips=pool)));tmp.replace(dest)
if world==1:
 from lip.data.sampling import merge_sampling
 merge_sampling(index,a.length,1,a.test_finalized)
