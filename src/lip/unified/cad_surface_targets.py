"""Teacher-only CAD patch correspondence, with declared BOP symmetry orbits."""
from dataclasses import replace
import torch
from .cad_surface_cache import surface_tokens


def orbit_distance(points,anchors,metadata,center,diameter):
    """Squared distance to the discrete/analytic-continuous symmetry orbit."""
    transforms=[torch.eye(4,device=points.device)]
    transforms += [torch.as_tensor(x,device=points.device,dtype=torch.float32).reshape(4,4) for x in metadata.get('symmetries_discrete',[])]
    continuous=metadata.get('symmetries_continuous',[])
    if len(continuous)>1:raise ValueError('Multiple continuous symmetry axes unsupported')
    best=None
    for transform in transforms:
        rotation=transform[:3,:3];translation=transform[:3,3]/1000
        transformed=((anchors*diameter+center)@rotation.T+translation-center)/diameter
        if continuous:
            axis=points.new_tensor(continuous[0]['axis']);axis=axis/axis.norm()
            origin=(points.new_tensor(continuous[0]['offset'])/1000-center)/diameter
            a=points-origin;b=transformed-origin;za=a@axis;zb=b@axis
            ra=(a-za[:,None]*axis).norm(dim=-1);rb=(b-zb[:,None]*axis).norm(dim=-1)
            distance=(za[:,None]-zb[None]).square()+(ra[:,None]-rb[None]).square()
        else:distance=(points[:,None]-transformed[None]).square().sum(-1)
        best=distance if best is None else torch.minimum(best,distance)
    return best


@torch.no_grad()
@torch.autocast('cuda',enabled=False)
def surface_targets(model,scenes,target):
    rows={'real':[],'proxy':[]};weights={'real':[],'proxy':[]}
    sample=torch.tensor([1,5,8,12],device=target.surface_xyz.device)
    for lane,scene in enumerate(scenes):
        bank=surface_tokens(scene.cad,model.cad_surface_cache,model.cad_surface_count)
        assets=[p for p in scene.cad['_asset_files'] if p.suffix=='.obj']
        if len(assets)!=1:raise ValueError('Unique CAD asset required for symmetry identity')
        name=assets[0].parent.name
        if name not in model.cad_surface_symmetry:raise ValueError('Missing explicit symmetry metadata for '+name)
        xyz=target.surface_xyz[lane].reshape(3,16,14,16,14).permute(1,3,2,4,0)
        xyz=xyz[:,:,sample][:,:,:,sample].reshape(256,16,3)
        distance=orbit_distance(xyz.reshape(-1,3),bank['coord'].float(),model.cad_surface_symmetry[name],scene.center.float(),scene.diameter)
        plausible=distance.amin(-1)<.1**2
        assignment=(-distance/(2*.03**2)).softmax(-1).reshape(256,16,-1)
        for kind,mask in [('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight)]:
            selected=mask[lane,0].reshape(16,14,16,14).permute(0,2,1,3)
            selected=selected[:,:,sample][:,:,:,sample].reshape(256,16)&plausible.reshape(256,16)
            mass=selected.float().mean(-1)
            distribution=(assignment*selected[:,:,None]).sum(1)/selected.sum(1).clamp_min(1)[:,None]
            rows[kind].append(torch.nn.functional.pad(distribution,(0,1)))
            weights[kind].append(mass*(mass>=.5))
    return replace(target,cad_surface_target_real=torch.stack(rows['real']),cad_surface_target_proxy=torch.stack(rows['proxy']),
                   cad_surface_weight_real=torch.stack(weights['real']),cad_surface_weight_proxy=torch.stack(weights['proxy']))


def surface_correspondence_loss(output,target):
    from lip.jepa.losses import sample_mean
    if target.cad_surface_target_real is None:raise ValueError('Local CAD correspondence targets missing')
    logs=output['cad_match_log_prob'].float();available=output['cad_surface_available']
    mask=torch.nn.functional.pad(available,(0,1),value=False);values=[]
    for truth,weight in [(target.cad_surface_target_real,target.cad_surface_weight_real),
                         (target.cad_surface_target_proxy,target.cad_surface_weight_proxy)]:
        truth=truth.detach()*mask[:,None];mass=truth.sum(-1)
        truth=truth/mass.clamp_min(1e-8)[...,None]
        values.append(sample_mean(-(truth*logs).sum(-1),weight*(mass>0))[0])
    return values[0]+.5*values[1],torch.stack([x.detach() for x in values])


@torch.no_grad()
def correspondence_score(output,target,proxy=False):
    if 'cad_match_log_prob' not in output:return None
    truth=target.cad_surface_target_proxy if proxy else target.cad_surface_target_real
    weight=target.cad_surface_weight_proxy if proxy else target.cad_surface_weight_real
    mask=torch.nn.functional.pad(output['cad_surface_available'],(0,1),value=False)
    truth=truth*mask[:,None];mass=truth.sum(-1);weight=weight*(mass>0)
    if not weight.any():return None
    truth=truth/mass.clamp_min(1e-8)[...,None];logs=output['cad_match_log_prob'].float()
    prediction=logs.argmax(-1);selected=truth.gather(-1,prediction[...,None]).squeeze(-1)
    nll=-(truth*logs).sum(-1);entropy=-(truth*truth.clamp_min(1e-8).log()).sum(-1)
    mean=lambda x:float((x*weight).sum()/weight.sum())
    return dict(patches=float(weight.sum()),cross_entropy=mean(nll),teacher_entropy=mean(entropy),
        excess_kl=mean(nll-entropy),top1_supported=mean((selected>=.5*truth.amax(-1)).float()),
        null_rate=mean((prediction==truth.shape[-1]-1).float()))
