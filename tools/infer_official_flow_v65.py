"""Flow-only worker: reads student inputs only, never GT/label files."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from official_flow_adapter_v65 import load,ROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.cuda.set_device(0);model=load()
 cache=ROOT/'experiment/cache'/f'rank{a.rank}';out=ROOT/'experiment/predictions'/f'rank{a.rank}';out.mkdir(parents=True,exist_ok=False)
 rows=[json.loads(s) for s in (cache/'frames.jsonl').read_text().splitlines()]
 with torch.no_grad():
  for i,row in enumerate(rows):
   path=cache/row['input_file'];assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==row['input_sha256']
   x=torch.load(path,weights_only=True);assert set(x)=={'query','template','mask'}
   q=F.interpolate(x['query'].cuda(),size=(280,280),mode='bilinear',align_corners=False)
   mask=F.interpolate(x['mask'][:,None].float().cuda(),size=(280,280),mode='nearest')[:,0]
   result={}
   for name in ('shared_rgb','official_gray'):
    template=x['template'].cuda()
    if name=='official_gray':template=torch.where(x['mask'][:,None].cuda(),template,.5)
    t=F.interpolate(template,size=(280,280),mode='bilinear',align_corners=False)
    torch.cuda.synchronize();start=time.monotonic();flow,confidence=model(q,t,mask);torch.cuda.synchronize()
    assert torch.isfinite(flow).all() and torch.isfinite(confidence).all()
    result[name]=dict(flow=flow.cpu(),confidence=confidence.cpu(),mask=mask.cpu(),seconds=time.monotonic()-start)
   torch.save(result,out/f'prediction{i:03d}.pt')
 (out/'receipt.json').write_text(json.dumps(dict(completed=True,frames=len(rows),strict_weight_load=True,teacher_input=False,
  pose_solver_called=False,precision='float32',student_input_bytes_verified=True,
  preprocessing='224->280 bilinear align_corners=False RGB; nearest template mask; shared RGB and official0.5 background variants'),indent=2)+'\n')

if __name__=='__main__':main()
