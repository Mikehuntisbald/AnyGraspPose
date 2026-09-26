"""Complete CAD point retrieval from JEPA/DPT queries, with explicit labels.

Forward XYZ is an actual cached CAD surface point. A local straight-through
surrogate carries geometric gradients; global correspondence supervision is
separate and never chooses the forward index. No sensor/encoder pose bypass.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CADAtlasDecoder(nn.Module):
    def __init__(self,feature_dim,width=32,supervised_prior=True):
        super().__init__()
        self.query=nn.Sequential(nn.Conv2d(16,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,1))
        self.descriptor=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,width))
        self.geometry=nn.Sequential(nn.Linear(21,64),nn.GELU(),nn.Linear(64,width))
        self.prior_sigma=.1
        self.temperature=.1
        self.supervised_prior=supervised_prior

    def scores(self,query,keys,prior,xyz,available):
        logits=(query@keys.transpose(-1,-2))/self.temperature
        distance=(prior.square().sum(-1,keepdim=True)+xyz.square().sum(-1)[:,None]-2*(prior@xyz.transpose(-1,-2))).clamp_min(0)
        return (logits-distance/(2*self.prior_sigma**2)).masked_fill(~available[:,None],-1e4)

    def forward(self,dense,fallback,features,geometry,available):
        safe_features=torch.where(available[...,None],features,0.)
        safe_geometry=torch.where(available[...,None],geometry,0.)
        query=F.normalize(self.query(dense).flatten(2).transpose(1,2).float(),dim=-1)
        keys=F.normalize((self.descriptor(safe_features)+self.geometry(safe_geometry)).float(),dim=-1)
        xyz=safe_geometry[:,:,:3].float()
        prior=fallback[:,:3].detach().float().flatten(2).transpose(1,2)
        # The full bank search is hard and teacher-independent. Recompute only
        # a small local neighborhood with gradients; global CE is computed on
        # fixed supervised query samples by atlas_correspondence_loss below.
        indices=[]
        with torch.no_grad(),torch.autocast(dense.device.type,enabled=False):
            for start in range(0,query.shape[1],2048):
                scores=self.scores(query[:,start:start+2048].detach(),keys.detach(),prior[:,start:start+2048],xyz,available)
                indices.append(scores.topk(min(8,xyz.shape[1]),dim=-1).indices)
        selected=torch.cat(indices,1)
        batch=torch.arange(len(query),device=query.device)[:,None,None]
        with torch.autocast(dense.device.type,enabled=False):
            selected_xyz=xyz[batch,selected]
            selected_keys=keys[batch,selected]
            local=(query[:,:,None]*selected_keys).sum(-1)/self.temperature
            local=local-(prior[:,:,None]-selected_xyz).square().sum(-1)/(2*self.prior_sigma**2)
            local=local.masked_fill(~available[batch,selected],-1e4)
            probability=local.softmax(-1)
            soft=(probability[...,None]*selected_xyz).sum(2)
            hard=selected_xyz[:,:,0]
            recovered=hard+(soft-soft.detach())
            recovered=recovered.transpose(1,2).reshape_as(fallback[:,:3])
            recovered=torch.where(available.any(-1)[:,None,None,None],recovered,fallback[:,:3].float())
        surface=torch.cat((recovered,fallback[:,3:].float()),1)
        output=dict(atlas_query=query,atlas_keys=keys,atlas_xyz=xyz,atlas_geometry=safe_geometry,atlas_available=available,
                            atlas_prior=prior,atlas_index=selected[:,:,0],atlas_fallback=fallback,
                            atlas_temperature=self.temperature,atlas_prior_sigma=self.prior_sigma,
                            atlas_supervised_prior=self.supervised_prior)
        if hasattr(self,'image_readout'):
            output.update(self.image_readout(output,dense.shape[-2:]))
        return surface,output


@torch.autocast('cuda',enabled=False)
def atlas_correspondence_loss(output,target,visible,max_queries=256):
    """Loss-only nearest8 soft labels in the fixed canonical texture gauge.

Each real/proxy/visible source is normalized separately, then each eligible
example equally. Labels use CAD canonical XYZ; real depth is unchanged.
"""
    q,k=output['atlas_query'].float(),output['atlas_keys'].float()
    xyz=output['atlas_xyz'].float();available=output['atlas_available']
    prior=output['atlas_prior'].float();truth=target.cad_geometry_xyz.detach().float().flatten(2).transpose(1,2)
    total=q.sum()*0;metrics={}
    for name,mask,factor in [('real',target.geometry_real_weight,1.),('proxy',target.geometry_proxy_weight,.5),('visible',visible,.5)]:
        mask=mask&target.cad_geometry_valid
        losses=[];distances=[];correct=[]
        for b in range(len(q)):
            ids=mask[b].flatten().nonzero().flatten()
            if not len(ids) or not available[b].any():continue
            if len(ids)>max_queries:ids=ids[torch.linspace(0,len(ids)-1,max_queries,device=ids.device).long()]
            logits=q[b,ids]@k[b].T/output['atlas_temperature']
            if output.get('atlas_supervised_prior',True):
                offset=prior[b,ids,None]-xyz[b,None]
                logits=logits-offset.square().sum(-1)/(2*output['atlas_prior_sigma']**2)
            logits=logits.masked_fill(~available[b,None],-1e4)
            with torch.no_grad():
                distance=torch.cdist(truth[b,ids][None],xyz[b][None],compute_mode='donot_use_mm_for_euclid_dist')[0]
                distance=distance.masked_fill(~available[b,None],float('inf'))
                value,labels=distance.topk(min(8,xyz.shape[1]),largest=False)
                weights=(-value.square()/(2*.01**2)).softmax(-1)
            log_probability=logits.log_softmax(-1)
            losses.append(-(log_probability.gather(1,labels)*weights).sum(-1).mean())
            predicted=logits.detach().argmax(-1)
            distances.append(distance.gather(1,predicted[:,None]).mean())
            correct.append((predicted[:,None]==labels).any(-1).float().mean())
        if losses:
            total=total+factor*torch.stack(losses).mean()
            metrics['atlas_'+name+'_ce']=torch.stack(losses).mean().detach()
            metrics['atlas_'+name+'_distance_d']=torch.stack(distances).mean().detach()
            metrics['atlas_'+name+'_top8']=torch.stack(correct).mean().detach()
    return total,metrics
