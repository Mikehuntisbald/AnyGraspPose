"""Estimated-pose geometry relations; no teacher/GT inputs or encoder calls."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def dynamic_relations(scenes,camera,depth,render_camera,render_depth,bounds,base,diameter,cad_enabled):
    scale=diameter[:,None,None,None]
    ov=(depth>0)&bounds;rv=(render_depth>0)&bounds;both=ov&rv
    observed=(camera.permute(0,3,1,2)-base[:,:3,3,None,None])/scale
    rendered=(render_camera.permute(0,3,1,2)-base[:,:3,3,None,None])/scale
    difference=(depth-render_depth)/scale
    pool=lambda x:F.avg_pool2d(x.float(),14,14).flatten(2).transpose(1,2)
    mean=lambda x,m:pool(torch.where(m,x,0.))/pool(m).clamp_min(1e-6)
    dense=torch.cat((mean(observed.clamp(-4,4),ov),mean(rendered.clamp(-4,4),rv),
        mean((observed-rendered).clamp(-4,4),both),pool(ov),pool(rv),pool(both),
        pool(both&(difference<-.02)),pool(both&(difference.abs()<=.02)),pool(both&(difference>.02))),-1)
    # CAD reference dropout removes every CAD-conditioned dense component.
    dense=torch.cat((dense[:,:,:3],dense[:,:,3:9]*cad_enabled[:,None,None],dense[:,:,9:10],
                     dense[:,:,10:]*cad_enabled[:,None,None]),-1)
    cad=[]
    for i,s in enumerate(scenes):
        idx=torch.linspace(0,len(s.cad['coord'])-1,128,device=depth.device).long()
        points=(s.cad['coord'][idx]*s.diameter)@s.pose[:3,:3].T+s.pose[:3,3]
        normal=s.cad['normal'][idx]@s.pose[:3,:3].T
        projected=points@s.k_crop.T;uv=projected[:,:2]/projected[:,2:].clamp_min(.001)
        grid=((uv+.5)/112-1)[None,None]
        sample=lambda x:F.grid_sample(x[i:i+1].float(),grid,mode='nearest',align_corners=False)[0,0,0]
        od=sample(depth);rd=sample(render_depth)
        inside=(uv>=0).all(-1)&(uv<=223).all(-1)&(points[:,2]>.001)
        front=inside&(rd>0)&((rd-points[:,2]).abs()<(.002+.01*s.diameter))
        usable=front&(od>0)
        delta=(od-points[:,2])/s.diameter
        relation=torch.cat(((points-s.pose[:3,3])/s.diameter,normal,uv/112-1,
            (points[:,2]/s.diameter)[:,None],torch.where(usable,delta.clamp(-2,2),0.)[:,None],
            (inside&(od>0))[:,None],front[:,None],(usable&(delta<-.02))[:,None],
            (usable&(delta.abs()<=.02))[:,None],(usable&(delta>.02))[:,None]),-1)
        cad.append(relation)
    return dense,torch.stack(cad)*cad_enabled[:,None,None]
