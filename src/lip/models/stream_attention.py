"""Frame-block SDPA with query-only readouts and a full-prefix correctness oracle."""
import torch
from torch import nn
from torch.nn import functional as F
from lip.engine.stream_state import FrameMeta, KVBlock, TemporalCache


class StreamLayer(nn.Module):
    # Names intentionally match TransformerEncoderLayer for audited migration.
    def __init__(self,dropout=0.):
        super().__init__()
        self.self_attn=nn.MultiheadAttention(256,8,dropout=dropout,batch_first=True)
        self.linear1=nn.Linear(256,1024);self.linear2=nn.Linear(1024,256)
        self.norm1=nn.LayerNorm(256);self.norm2=nn.LayerNorm(256)
        self.dropout=nn.Dropout(dropout);self.dropout1=nn.Dropout(dropout);self.dropout2=nn.Dropout(dropout)

    @staticmethod
    def heads(x):
        return x.reshape(x.shape[0],-1,8,32).transpose(1,2)

    def project(self,source,readout):
        ns=self.norm1(source);nr=self.norm1(readout)
        w=self.self_attn.in_proj_weight;b=self.self_attn.in_proj_bias
        q=self.heads(F.linear(torch.cat((ns,nr),1),w[:256],b[:256]))
        k,v=F.linear(ns,w[256:],b[256:]).chunk(2,-1)
        return q,self.heads(k),self.heads(v)

    def residual(self,x,q,k,v,bias):
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,
            dropout_p=self.self_attn.dropout if self.training else 0.,is_causal=False)
        y=y.transpose(1,2).reshape(x.shape)
        x=x+self.dropout1(self.self_attn.out_proj(y))
        return x+self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))


class StreamTemporal(nn.Module):
    def __init__(self,memory_frames=8,dropout=0.,time_unit=1/30):
        super().__init__();self.memory_frames=int(memory_frames);self.time_unit=float(time_unit)
        if self.memory_frames<1 or self.time_unit<=0:raise ValueError('Invalid memory/time scale')
        self.layers=nn.ModuleList([StreamLayer(dropout) for _ in range(4)])
        self.norm=nn.LayerNorm(256)
        self.time_bias=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,8))
        nn.init.zeros_(self.time_bias[-1].weight);nn.init.zeros_(self.time_bias[-1].bias)

    def bias(self,queries,keys,readouts,object_readout):
        # Query ordering: all source frames, then all frame-local readouts.
        qid=torch.stack([m.frame_id for m in queries],1)
        qtag=torch.stack([m.stream_tag for m in queries],1)
        qt=torch.stack([m.timestamp for m in queries],1)
        kid=torch.stack([m.frame_id for m in keys],1)
        ktag=torch.stack([m.stream_tag for m in keys],1)
        kt=torch.stack([m.timestamp for m in keys],1)
        repeat_q=lambda x:torch.cat((x.repeat_interleave(17,1),x.repeat_interleave(readouts,1)),1)
        delta=repeat_q(qid)[:,:,None]-kid.repeat_interleave(17,1)[:,None,:]
        valid=torch.cat([m.key_valid for m in keys],1)
        allowed=(delta>=0)&(delta<self.memory_frames)&(repeat_q(qtag)[:,:,None]==ktag.repeat_interleave(17,1)[:,None,:])&valid[:,None,:]
        # Relative time is formed in float64 before narrowing. No cached token is reencoded.
        elapsed=(repeat_q(qt)[:,:,None]-kt.repeat_interleave(17,1)[:,None,:]).clamp_min(0)
        with torch.autocast(qt.device.type,enabled=False):
            tb=self.time_bias(torch.log1p(elapsed/self.time_unit).float()[...,None]).permute(0,3,1,2)
        if object_readout:
            role=torch.cat([m.role_bias for m in keys],1)
            selector=tb.new_zeros(tb.shape[2]);selector[len(queries)*17::readouts]=1
            tb=tb+selector[None,None,:,None]*role[:,None,None,:]
        # All-invalid padding lanes receive a zero-valued fallback key, then their
        # queries are zeroed. Valid source values are separately sanitized below.
        empty=~allowed.any(-1)
        fallback=torch.zeros_like(allowed);fallback[:,:,0]=empty
        return tb.masked_fill(~(allowed|fallback)[:,None],float('-inf'))

    @staticmethod
    def sanitize(x,valid):
        return torch.where(valid[...,None],x,torch.zeros_like(x))

    def forward(self,source,readout,meta:FrameMeta,cache=None,object_readout=False):
        if source.shape[1:]!=(17,256):raise ValueError('Expected [B,17,256] source tokens')
        cache=cache or TemporalCache(capacity=self.memory_frames)
        if cache.capacity!=self.memory_frames:raise ValueError('Reset after changing cache capacity')
        n=max(0,self.memory_frames-1)
        pm=cache.metadata[-n:] if n else ()
        metas=pm+(meta,);bias=self.bias([meta],metas,readout.shape[1],object_readout)
        source=self.sanitize(source,meta.key_valid)
        qvalid=torch.cat((meta.key_valid,meta.key_valid.any(-1,keepdim=True).expand(-1,readout.shape[1])),1)
        new_layers=[]
        for l,layer in enumerate(self.layers):
            source=self.sanitize(source,meta.key_valid)
            q,k,v=layer.project(source,readout)
            mask=meta.key_valid[:,None,:,None]
            k=torch.where(mask,k,0.);v=torch.where(mask,v,0.)
            previous=cache.layers[l][-n:] if n else ()
            block=KVBlock(k,v);blocks=previous+(block,)
            keys=torch.cat([x.key for x in blocks],2);values=torch.cat([x.value for x in blocks],2)
            x=layer.residual(torch.cat((source,readout),1),q,keys,values,bias)
            x=self.sanitize(x,qvalid);source,readout=x[:,:17],x[:,17:]
            new_layers.append((cache.layers[l]+(block,))[-self.memory_frames:])
        next_cache=TemporalCache(tuple(new_layers),(cache.metadata+(meta,))[-self.memory_frames:],self.memory_frames)
        z=self.norm(readout);z=self.sanitize(z,qvalid[:,17:])
        return z,next_cache

    def full_reference(self,sources,readouts,metadata,object_readout=False):
        """Full immutable prefix at EVERY layer with local frame mask, including eviction.

        sources [B,F,17,256], readouts [B,F,R,256]. Old sources are not recropped.
        This deliberately slow path is not the production speed baseline.
        """
        b,f,s,d=sources.shape;r=readouts.shape[2]
        if s!=17 or f!=len(metadata):raise ValueError('Source / metadata mismatch')
        valid=torch.cat([m.key_valid for m in metadata],1)
        rv=torch.stack([m.key_valid.any(-1) for m in metadata],1).repeat_interleave(r,1)
        qvalid=torch.cat((valid,rv),1)
        source=self.sanitize(sources.reshape(b,f*s,d),valid);readout=readouts.reshape(b,f*r,d)
        bias=self.bias(metadata,metadata,r,object_readout)
        projected=[]
        for layer in self.layers:
            source=self.sanitize(source,valid)
            q,k,v=layer.project(source,readout)
            mask=valid[:,None,:,None];k=torch.where(mask,k,0.);v=torch.where(mask,v,0.)
            projected.append(KVBlock(k,v))
            x=layer.residual(torch.cat((source,readout),1),q,k,v,bias)
            x=self.sanitize(x,qvalid);source,readout=x[:,:f*s],x[:,f*s:]
        return self.sanitize(self.norm(readout),rv).reshape(b,f,r,d),projected
