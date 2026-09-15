"""Train-only visible CAD-to-observation correspondence targets; no hand/mask labels."""
import torch
from torch.nn import functional as F


def sample_nearest(image,uv):
    h,w=image.shape[-2:]
    grid=torch.stack((2*(uv[...,0]+.5)/w-1,2*(uv[...,1]+.5)/h-1),-1)
    return F.grid_sample(image,grid[:,None],mode='nearest',padding_mode='zeros',align_corners=False)[:,:,0].transpose(1,2)


@torch.no_grad()
def correspondence_targets(geometry,diameter,target,k_crop,crop_matrix,observed_depth,gt_render_depth,eligible):
    """B x observation-query x CAD-key distribution, and supported query mask.

    CAD token representatives are nearest samples at the 14x14 cell centers.
    Each point's true current projection splats onto at most four query cells.
    Visibility requires both GT object self-visibility and observed RGB-D support.
    All lengths are meters, XYZ is centered object coordinates, pixels use the
    existing OpenCV integer-center / align_corners=False convention.
    """
    if geometry.shape[1:]!=(9,224,224):raise ValueError('Expected native 224 geometry with nine channels')
    b=len(geometry);device=geometry.device
    with torch.autocast(device.type,enabled=False):
        geometry=geometry.float();diameter=diameter.float();target=target.float()
        axis=(torch.arange(14,device=device,dtype=torch.float32)+.5)*16-.5
        yy,xx=torch.meshgrid(axis,axis,indexing='ij');pixels=torch.stack((xx,yy),-1).reshape(1,196,2).expand(b,-1,-1)
        source=sample_nearest(geometry[:,3:7],pixels)
        xyz=source[:,:,1:]*diameter[:,None,None]
        camera=xyz@target[:,:3,:3].transpose(-1,-2)+target[:,None,:3,3]
        project=camera@k_crop.float().transpose(-1,-2)
        uv=project[:,:,:2]/project[:,:,2:].clamp_min(1e-6)
        gt_z=sample_nearest(gt_render_depth.float(),uv)[:,:,0]
        hom=torch.cat((uv,torch.ones_like(uv[:,:,:1])),-1)@torch.linalg.inv(crop_matrix.float()).transpose(-1,-2)
        raw_uv=hom[:,:,:2]/hom[:,:,2:].clamp_min(1e-6)
        obs_z=sample_nearest(observed_depth.float(),raw_uv)[:,:,0]
        z=camera[:,:,2]
        gt_tol=torch.maximum(diameter*.01,diameter.new_full((b,),.002))[:,None]
        obs_tol=torch.maximum(diameter*.02,diameter.new_full((b,),.005))[:,None]
        visible=(source[:,:,0]>.5)&(z>.001)&torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=223).all(-1)
        visible &= (gt_z>0)&((z-gt_z).abs()<=gt_tol)&(obs_z>0)&((z-obs_z).abs()<=obs_tol)&eligible[:,None]
        # Forward splatting allows multiple CAD keys to share an observation
        # query. Row normalization yields a soft, multi-positive target.
        cell=(uv+.5)/16-.5;lo=cell.floor();fraction=cell-lo
        distribution=geometry.new_zeros(b,196,196)
        batch=torch.arange(b,device=device)[:,None].expand(b,196)
        key=torch.arange(196,device=device)[None].expand(b,196)
        for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
            x=lo[:,:,0].long()+dx;y=lo[:,:,1].long()+dy
            weight=(fraction[:,:,0] if dx else 1-fraction[:,:,0])*(fraction[:,:,1] if dy else 1-fraction[:,:,1])
            valid=visible&(x>=0)&(x<14)&(y>=0)&(y<14)
            distribution.index_put_((batch[valid],(y*14+x)[valid],key[valid]),weight[valid],accumulate=True)
        mass=distribution.sum(-1);supported=mass>1e-6
        distribution=distribution/mass.clamp_min(1e-6)[:,:,None]
        return distribution,supported,dict(visible_keys=visible.sum(),supported_queries=supported.sum(),eligible_clips=eligible.sum())


def spatial_alignment_loss(log_probability,distribution,supported):
    count=supported.sum()
    loss=-(distribution*log_probability).sum(-1)
    mean=(loss*supported).sum()/count.clamp_min(1)
    predicted=log_probability.detach().argmax(-1)
    hit=distribution.gather(-1,predicted[...,None])[:,:,0]>0
    accuracy=(hit&supported).sum().float()/count.clamp_min(1)
    return mean,accuracy
