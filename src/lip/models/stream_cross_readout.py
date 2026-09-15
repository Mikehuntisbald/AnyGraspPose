"""Object queries causal context history; a learned gate scales its residual."""
import torch
from torch import nn
from torch.nn import functional as F
from lip.engine.stream_state import ContextBlock


class StreamCrossReadout(nn.Module):
    def __init__(self,memory_frames=8):
        super().__init__();self.dual=True;self.memory_frames=memory_frames
        self.query=nn.Parameter(torch.empty(1,2,256));nn.init.normal_(self.query,std=.02)
        self.cross_attn=nn.MultiheadAttention(256,8,dropout=0.,batch_first=True)
        # Starts at raw object latent, not at the removed legacy linear fusion.
        nn.init.zeros_(self.cross_attn.out_proj.weight);nn.init.zeros_(self.cross_attn.out_proj.bias)
        self.gate=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-2.)

    @staticmethod
    def heads(x):return x.reshape(len(x),-1,8,32).transpose(1,2)

    def project_context(self,context,metadata):
        w=self.cross_attn.in_proj_weight;b=self.cross_attn.in_proj_bias
        valid=metadata.key_valid.any(-1)[:,None,None]
        context=torch.where(valid,context,0.)
        k,v=F.linear(context,w[256:],b[256:]).chunk(2,-1)
        mask=valid[:,None]
        return ContextBlock(torch.where(mask,self.heads(k),0.),torch.where(mask,self.heads(v),0.),metadata)

    def attend(self,objects,query_metadata,blocks):
        w=self.cross_attn.in_proj_weight;b=self.cross_attn.in_proj_bias
        q=self.heads(F.linear(objects,w[:256],b[:256]))
        qid=torch.stack([m.frame_id for m in query_metadata],1)
        qt=torch.stack([m.timestamp for m in query_metadata],1)
        tag=torch.stack([m.stream_tag for m in query_metadata],1)
        kid=torch.stack([b.metadata.frame_id for b in blocks],1)
        kt=torch.stack([b.metadata.timestamp for b in blocks],1)
        ktag=torch.stack([b.metadata.stream_tag for b in blocks],1)
        valid=torch.stack([b.metadata.key_valid.any(-1) for b in blocks],1)
        delta=qid[:,:,None]-kid[:,None,:]
        allowed=(delta>=0)&(delta<self.memory_frames)&(qt[:,:,None]>=kt[:,None,:])&(tag[:,:,None]==ktag[:,None,:])&valid[:,None,:]
        qvalid=torch.stack([m.key_valid.any(-1) for m in query_metadata],1)
        has_key=allowed.any(-1)&qvalid
        fallback=torch.zeros_like(allowed);fallback[:,:,0]=~allowed.any(-1)
        bias=torch.zeros_like(allowed,dtype=q.dtype).masked_fill(~(allowed|fallback),float('-inf'))
        h=F.scaled_dot_product_attention(q,torch.cat([b.key for b in blocks],2),torch.cat([b.value for b in blocks],2),
                                         attn_mask=bias[:,None],dropout_p=0.,is_causal=False)
        h=self.cross_attn.out_proj(h.transpose(1,2).reshape_as(objects))
        h=torch.where(has_key[...,None],h,0.)
        # One O and H token per frame: pooling is the identity.
        gate=torch.sigmoid(getattr(self,'cross_gate',self.gate)(torch.cat((objects,h),-1)))
        updated=torch.where(qvalid[...,None],objects+gate*h,0.)
        return updated,h,gate

    def forward(self,z,metadata,previous=()):
        n=max(0,self.memory_frames-1);previous=previous[-n:] if n else ()
        obj,ctx=z[:,:1],z[:,1:2]
        block=self.project_context(ctx,metadata);blocks=previous+(block,)
        updated,h,gate=self.attend(obj,[metadata],blocks)
        return dict(latent=updated[:,0],latent_object=obj[:,0],latent_context=ctx[:,0],
                    context_gate=gate[:,0],cross_attention_output=h[:,0]),blocks

    def full_reference(self,z,metadata):
        """All prefix contexts with an explicit causal/local/identity mask."""
        blocks=tuple(self.project_context(z[:,i,1:2],m) for i,m in enumerate(metadata))
        return self.attend(z[:,:,0],metadata,blocks)


class StreamResidualCrossReadout(StreamCrossReadout):
    """Keep the learned dual function, then add an initially zero cross update."""
    def __init__(self,memory_frames=8):
        super().__init__(memory_frames)
        from lip.models.stream_readout import StreamReadout
        parent=StreamReadout(dual=True)
        self.cross_gate=self.gate
        self.gate=parent.gate
        self.context_projection=parent.context_projection

    def parent_latent(self,objects,contexts):
        gate=torch.sigmoid(self.gate(torch.cat((objects,contexts),-1)))
        return objects+gate*self.context_projection(contexts)

    def forward(self,z,metadata,previous=()):
        result,blocks=super().forward(z,metadata,previous)
        base=self.parent_latent(z[:,0],z[:,1])
        result['latent_parent']=base
        result['latent']=torch.where(metadata.key_valid.any(-1)[:,None],base+result['context_gate']*result['cross_attention_output'],0.)
        return result,blocks

    def full_reference(self,z,metadata):
        _,h,gate=super().full_reference(z,metadata)
        base=self.parent_latent(z[:,:,0],z[:,:,1])
        valid=torch.stack([m.key_valid.any(-1) for m in metadata],1)
        return torch.where(valid[...,None],base+gate*h,0.),h,gate
