"""Frozen normal diagnostics: separate invalid geometry from unit-vector angles."""
import math
import torch
from .surface_normals import normal_terms, normal_field


@torch.no_grad()
def audit_normals(prediction, target, mask):
    results={}
    with torch.autocast(prediction.device.type,enabled=False):
        p=prediction.float();t=target.float()
        for stride in (1,2):
            _,eligible,legacy_cos=normal_terms(p,t,mask,stride)
            def vectors(x):
                dx=x[:,:,stride:-stride,2*stride:]-x[:,:,stride:-stride,:-2*stride]
                dy=x[:,:,2*stride:,stride:-stride]-x[:,:,:-2*stride,stride:-stride]
                lx=dx.norm(dim=1,keepdim=True);ly=dy.norm(dim=1,keepdim=True)
                cross=torch.linalg.cross(dx,dy,dim=1);area=cross.norm(dim=1,keepdim=True)
                return cross/area.clamp_min(1e-20),lx,ly,area/(lx*ly).clamp_min(1e-20)
            unit,lx,ly,sine=vectors(p.nan_to_num());truth,_,_,_=vectors(t.nan_to_num())
            finite=torch.isfinite(p).all(1,keepdim=True)
            pf=finite[:,:,stride:-stride,stride:-stride].clone()
            for m in (finite[:,:,stride:-stride,:-2*stride],finite[:,:,stride:-stride,2*stride:],
                      finite[:,:,:-2*stride,stride:-stride],finite[:,:,2*stride:,stride:-stride]):pf &= m
            cosine=(unit*truth).sum(1,keepdim=True).clamp(-1,1)
            angle=cosine.acos()*180/math.pi;unoriented=cosine.abs().acos()*180/math.pi
            small=(lx<=1e-4)|(ly<=1e-4);collinear=sine<=.05
            good=pf&~small&~collinear
            legacy_norm=normal_field(p.nan_to_num(),stride)[0].norm(dim=1,keepdim=True)
            y,x=torch.meshgrid(torch.arange(stride,p.shape[-2]-stride,device=p.device),torch.arange(stride,p.shape[-1]-stride,device=p.device),indexing='ij')
            boundary=((x-stride)//14!=(x+stride)//14)|((y-stride)//14!=(y+stride)//14)
            for label,region in [('all',torch.ones_like(boundary)),('within_patch',~boundary),('cross_patch',boundary)]:
                use=eligible&region[None,None];valid=use&good
                values={'eligible':use.sum(),'degenerate':(use&~good).sum(),
                        'nonfinite':(use&~pf).sum(),'zero_tangent':(use&((lx==0)|(ly==0))).sum(),
                        'small_tangent':(use&small).sum(),'near_collinear':(use&collinear).sum(),
                        'legacy_attenuated':(use&(legacy_norm<.999)).sum(),'valid_unit':valid.sum(),
                        'unit_angle_sum':torch.where(valid,angle,0.).sum(),
                        'unoriented_angle_sum':torch.where(valid,unoriented,0.).sum(),
                        'negative_dot':(valid&(cosine<0)).sum(),
                        'near_opposite':(valid&(angle>=150)).sum(),
                        'near_aligned':(valid&(angle<=30)).sum(),
                        'legacy_angle_sum':torch.where(use,legacy_cos.acos()*180/math.pi,0.).sum()}
                for i in range(12):values[f'angle_bin_{i}']=(valid&(angle>=i*15)&((angle<(i+1)*15) if i<11 else (angle<=180))).sum()
                packed=torch.stack([v.double() for v in values.values()]).cpu().tolist()
                results[f's{stride}_{label}']=dict(zip(values,packed))
    return results
