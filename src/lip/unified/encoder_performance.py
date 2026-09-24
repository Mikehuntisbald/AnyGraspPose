"""Compile unchanged DINO math; explicit full-state execution-only migration."""
import copy
import types
import torch


def compiled_attention(self,x,attn_bias=None):
    if attn_bias is not None:raise ValueError('Dense fixed-shape DINO only')
    b,n,c=x.shape
    q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).unbind(2)
    value=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=0.)
    return self.proj_drop(self.proj(value.transpose(1,2).reshape(b,n,c)))


def compiled_features(backbone):
    def blocks(tokens):
        outputs=[]
        for i,block in enumerate(backbone.blocks):
            tokens=block(tokens)
            if i in (5,11):outputs.append(backbone.norm(tokens)[:,1+backbone.num_register_tokens:])
        return tuple(outputs)
    trunk=torch.compile(blocks,fullgraph=True,dynamic=False,
        options={'emulate_precision_casts':True,'force_same_precision':True})
    def features(rgb,n,reshape,norm):
        if n!=[5,11] or reshape or not norm:raise ValueError('Fixed DINO feature contract')
        # Keep the custom FP32 position interpolation transpose outside AOT's
        # BF16 backward autocast; all twelve transformer blocks are compiled.
        return trunk(backbone.prepare_tokens_with_masks(rgb))
    return features


def compile_encoders(model):
    for encoder in (model.encoder, getattr(model, 'ema_teacher', None)):
        if encoder is None: continue
        backbone=encoder.backbone
        # xFormers' custom unbind autograd cannot be traced in this environment.
        # Native SDPA preserves weights; explicit precision casts retain eager
        # intermediate BF16 rounding rather than eliding it during fusion.
        for block in backbone.blocks:
            block.attn.forward=types.MethodType(compiled_attention,block.attn)
        if hasattr(backbone,'_ema_position_matrices'):
            # Materialize immutable interpolation operator outside the compiled graph.
            with torch.no_grad():
                backbone.interpolate_pos_encoding(torch.zeros(1,257,384,device=backbone.pos_embed.device),224,224)
        object.__setattr__(encoder,'compiled_features',compiled_features(backbone))


def resume_performance(path,model,optimizer,scheduler,config,rank,world):
    from lip.engine.jepa_checkpoint import sha,load_core,restore_rng,core_state
    from lip.unified.checkpoint import software_environment
    if sha(path)!=config['performance_resume']['sha256']:
        raise ValueError('Performance source hash mismatch')
    source=torch.load(path,map_location='cpu',weights_only=False)
    expected=copy.deepcopy(source['config']);actual=copy.deepcopy(config)
    actual.pop('performance_resume');actual['runtime'].pop('compile_dino')
    actual['runtime'].pop('frame_batch',None)
    if actual['runtime'].pop('disable_history',False):
        assert actual['runtime']['history_pair_frames']==[] and actual['training']['history_pair_weight']==0
        actual['runtime']['history_pair_frames']=expected['runtime']['history_pair_frames']
        actual['training']['history_pair_weight']=expected['training']['history_pair_weight']
    actual['paths']['output']=expected['paths']['output']
    if actual!=expected:raise ValueError('Performance migration may only change execution flag and output path')
    assert source['software_environment']==software_environment()
    assert len(source['rng'])==world and source['step']==config['performance_resume']['step']
    assert source['sampler_position']==source['step']*config['training']['effective_batch']
    load_core(model,source['model']);optimizer.load_state_dict(source['optimizer']);scheduler.load_state_dict(source['scheduler'])
    restore_rng(source['rng'][rank])
    assert all(torch.equal(v.cpu(),source['model'][k]) for k,v in core_state(model).items())
    def exact(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
        if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
        return a==b
    assert exact(optimizer.state_dict(),source['optimizer']) and exact(scheduler.state_dict(),source['scheduler'])
    return source
