"""Execution-only optimizations; no parameters, targets, or optimizer changes."""
import torch
from torch.nn import functional as F

_GRIDS = {}


def pixel_grid(device, dtype, size=224):
    key=(str(device),dtype,size)
    if key not in _GRIDS:
        y,x=torch.meshgrid(torch.arange(size,device=device,dtype=dtype),
                           torch.arange(size,device=device,dtype=dtype),indexing='ij')
        _GRIDS[key]=torch.stack((x,y,torch.ones_like(x)),-1)
    return _GRIDS[key]


def camera_points_fast(depth,k):
    # Dataset calibration is validated before GPU training. inv_ex avoids the
    # mandatory CPU error check of inv; matrix arithmetic is unchanged.
    inverse=torch.linalg.inv_ex(k,check_errors=False).inverse
    return (pixel_grid(depth.device,torch.float32)@inverse.T)*depth[0,0,:,:,None]


def crop_images_fast(x,a,size=224,mode='bilinear'):
    h,w=x.shape[-2:]
    uv=pixel_grid(x.device,a.dtype,size)@torch.linalg.inv_ex(a,check_errors=False).inverse.T
    grid=torch.stack((2*(uv[...,0]+.5)/w-1,2*(uv[...,1]+.5)/h-1),-1)
    return F.grid_sample(x,grid[None].expand(len(x),-1,-1,-1),mode=mode,padding_mode='zeros',align_corners=False)


def crop_matrix_fast(vertices,base,k,size=224,expansion=2.):
    camera=vertices@base[:3,:3].T+base[:3,3];good=camera[:,2]>.001
    pixel=camera@k.T;uv=pixel[:,:2]/pixel[:,2:].clamp_min(1e-6)
    lo=uv.masked_fill(~good[:,None],float('inf')).amin(0)
    hi=uv.masked_fill(~good[:,None],-float('inf')).amax(0)
    any_good=good.any()
    lo=torch.where(any_good,lo,k[:2,2]);hi=torch.where(any_good,hi,k[:2,2])
    mid=(lo+hi)/2
    side=torch.where(any_good,((hi-lo).max()*expansion).clamp(64,4096),k.new_tensor(640.))
    scale=size/side;a=torch.eye(3,device=k.device,dtype=k.dtype)
    a[0,0]=a[1,1]=scale;a[:2,2]=(size-1)/2-scale*mid
    return a,a@k


class FrozenFeatureGraph:
    """Capture original eager DINO kernels, with private inputs/output copies.

    EMA parameters are updated in place: a replay reads the new values. Cached
    graph outputs are never cached teacher features. No attention substitution.
    """
    def __init__(self,encoder):
        if any(p.requires_grad for p in encoder.parameters()):
            raise ValueError('Forward-only graph cannot wrap a trainable encoder')
        self.encoder=encoder;self.cache={}

    def __call__(self,rgb,n,reshape,norm):
        if torch.is_grad_enabled():raise ValueError('Frozen feature graph requires no_grad')
        key=(tuple(rgb.shape),rgb.dtype,tuple(n),torch.is_autocast_enabled('cuda'),torch.get_autocast_dtype('cuda'))
        if key not in self.cache:
            static=rgb.detach().clone();stream=torch.cuda.Stream(device=rgb.device)
            stream.wait_stream(torch.cuda.current_stream(rgb.device))
            with torch.cuda.stream(stream):
                for _ in range(3):self.encoder.backbone.get_intermediate_layers(static,n=n,reshape=reshape,norm=norm)
            torch.cuda.current_stream(rgb.device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                outputs=self.encoder.backbone.get_intermediate_layers(static,n=n,reshape=reshape,norm=norm)
            self.cache[key]=(static,outputs,graph)
        static,outputs,graph=self.cache[key]
        static.copy_(rgb);graph.replay()
        return tuple(x.clone() for x in outputs)


def configure_execution(runner):
    flags=runner.config['runtime']
    for encoder,enabled in ((runner.reference_model.encoder,flags.get('reference_dino_graph',False)),
                            (runner.model.ema_teacher,flags.get('teacher_dino_graph',False))):
        if enabled and not hasattr(encoder,'graphed_features'):
            object.__setattr__(encoder,'graphed_features',FrozenFeatureGraph(encoder))
        elif not enabled and hasattr(encoder,'graphed_features'):
            delattr(encoder,'graphed_features')
    for model in (runner.model,runner.reference_model):
        model.fast_geometry=flags.get('fast_geometry',False)
        model.vector_geometry=flags.get('vector_geometry',False)


@torch.no_grad()
def prime_execution_graphs(runner):
    """Capture before background prefetch starts cudaHostAlloc/pin_memory.

    Global capture safety is retained; no relaxed/thread-local capture mode.
    All training shapes are static, and EMA values are read on each replay.
    """
    configure_execution(runner)
    c=runner.config;b=c['runtime']['microbatch'];chunk=c['runtime']['frame_batch']
    remaining=c['training']['episode_frames']-1
    counts={min(chunk,remaining)}
    if remaining%chunk:counts.add(remaining%chunk)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for encoder,batches in ((runner.reference_model.encoder,[2*b]),
                                (runner.model.ema_teacher,[2*b*t for t in sorted(counts)])):
            if hasattr(encoder,'graphed_features'):
                device=next(encoder.parameters()).device
                for batch in batches:encoder(torch.zeros(batch,3,224,224,device=device))


def geometry_inputs_fast(scenes,depth,bounds,base,diameter,cad_enabled):
    """Batch pointwise channels/pools; retain original per-view matrix products."""
    rendered=torch.stack([s.render['depth'] for s in scenes])
    rendered=torch.where(bounds,rendered,0.)
    xyz_render=torch.stack([s.render['xyz'] for s in scenes])
    d=diameter[:,None,None,None];z=base[:,2,3,None,None,None]
    valid=torch.isfinite(depth)&(depth>0);silhouette=rendered>0;both=valid&silhouette
    norm=lambda x,m:torch.where(m,x/d,0.).clamp(-2,2)
    geo=torch.cat((norm(depth-z,valid),valid.float(),norm(rendered-z,silhouette),silhouette.float(),
                   xyz_render/d,norm(depth-rendered,both),both.float()),1)
    geo[:,2:]*=cad_enabled[:,None,None,None]
    local=[];cad=[]
    for i,s in enumerate(scenes):
        camera=camera_points_fast(depth[i:i+1],s.k_crop)
        local.append(((camera-base[i,:3,3])@base[i,:3,:3]/diameter[i]).permute(2,0,1))
        features=s.cad['features'];key=(features.data_ptr(),features._version)
        if s.cad.get('_mean_identity')!=key:
            s.cad['_mean_features']=features.mean(0);s.cad['_mean_identity']=key
        cad.append(s.cad['_mean_features'])
    measured=depth>0;mass=F.avg_pool2d(measured.float(),14,14)
    pooled=F.avg_pool2d(torch.stack(local)*measured,14,14)/mass.clamp_min(1e-6)
    return geo,pooled.flatten(2).transpose(1,2),mass.flatten(1)>0,torch.stack(cad)
