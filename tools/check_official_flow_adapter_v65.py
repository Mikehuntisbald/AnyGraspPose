"""Exact official-method parity and render-only identity/translation controls."""
import ast,time,types,json
from pathlib import Path
import torch
from torch.nn import functional as F
from official_flow_adapter_v65 import load,ROOT,SOURCE

def main():
 torch.set_num_threads(2);m=load()
 tree=ast.parse((SOURCE/'model/gotrack.py').read_text())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GoTrack')
 functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('compute_flow','extract_features')]
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+functions,type_ignores=[])
 scope={'torch':torch,'time':time};exec(compile(ast.fix_missing_locations(module),'official_gotrack_methods','exec'),scope)
 m.opts=types.SimpleNamespace(frozen_backbone=True,crop_size=(280,280))
 m.extract_features=types.MethodType(scope['extract_features'],m)
 official=types.MethodType(scope['compute_flow'],m)
 rows=[]
 with torch.no_grad():
  for rank in range(8):
   x=torch.load(ROOT/'experiment/cache'/f'rank{rank}/input000.pt',weights_only=True)
   t=x['template'].cuda();mask=x['mask'][:,None].float().cuda()
   interior=-F.max_pool2d(-mask,5,1,2)>.999
   t280=F.interpolate(t,(280,280),mode='bilinear',align_corners=False)
   mask280=F.interpolate(mask,(280,280),mode='nearest')[:,0]
   for dx,dy in ((0,0),(7,-5)):
    q=torch.full_like(t,.5)
    sx0=max(-dx,0);sx1=min(224-dx,224);sy0=max(-dy,0);sy1=min(224-dy,224)
    q[:,:,sy0+dy:sy1+dy,sx0+dx:sx1+dx]=t[:,:,sy0:sy1,sx0:sx1]
    q280=F.interpolate(q,(280,280),mode='bilinear',align_corners=False)
    flow,confidence=m(q280,t280,mask280)
    if rank==0 and dx==0:
     ref=official(dict(rgbs_query=q280,rgbs_template=t280,masks_template=mask280))
     assert torch.equal(flow,ref['flows']) and torch.equal(confidence,ref['confidences'])
    mass=F.interpolate(mask280[:,None],(224,224),mode='bilinear',align_corners=False)
    restored=F.interpolate(flow,(224,224),mode='bilinear',align_corners=False)/mass.clamp_min(1e-8)/1.25
    valid=interior.clone();valid[:,:,:sy0]=False;valid[:,:,sy1:]=False;valid[:,:,:,:sx0]=False;valid[:,:,:,sx1:]=False
    error=(restored-restored.new_tensor([dx,dy])[None,:,None,None]).norm(dim=1,keepdim=True)
    rows.append(dict(rank=rank,dx=dx,dy=dy,pixels=int(valid.sum()),epe=float(error[valid].mean()),within3=float((error[valid]<=3).float().mean())))
 result=dict(official_compute_flow_bitwise_equal=True,official_source_methods_unmodified=True,teacher_used=False,
  scope='Eight fixed CAD renders; identity and exact7,-5px translation. Diagnostic only, not real-image performance.',rows=rows)
 (ROOT/'experiment/adapter_controls.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
