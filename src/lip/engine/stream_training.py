"""A whole burn-in/unroll is one DDP forward; pose feedback detaches, KV does not."""
import torch
from torch import nn
from lip.engine.stream_features import build_current_features,stack_current,mesh_to_device
from lip.engine.stream_state import FrameMeta
from lip.data.perturb import noisy_history
from lip.losses import pose_loss
from lip.geometry.so3 import angle


class StreamTrainingModule(nn.Module):
    def __init__(self,tracker,config,renderer):
        super().__init__();self.tracker=tracker;self.config=config;self.renderer=renderer

    def forward(self,samples,return_predictions=False):
        c=self.config;device=next(self.tracker.parameters()).device;burn=c['burn_in_frames'];unroll=c['supervised_unroll_frames']
        b=len(samples);meshes=[mesh_to_device(s['mesh'],device) for s in samples]
        accepted=torch.stack([s['initial_pose'].to(device).float() for s in samples]);previous=None
        if c['initial_pose_noise']:
            accepted=torch.stack([noisy_history(p[None],meshes[i]['diameter'],torch.Generator().manual_seed(samples[i]['sample']['seed']))[0][0] for i,p in enumerate(accepted)])
        times=torch.stack([s['timestamps'].to(device,dtype=torch.float64) for s in samples])
        k=[s['k'].to(device).float() for s in samples];cache=None;losses=[];metrics=[];predictions=[];cross_diagnostics=[]
        diameter=torch.stack([m['diameter'].float() for m in meshes]);points=torch.stack([m['points'].float() for m in meshes])
        batched=c.get('batch_current_features',False)
        if batched:
            from lip.engine.stream_batch_features import build_current_batch
            # Upload each raw fragment once. Images remain uint8/uint16 until
            # their current frame is consumed; future frames never enter features.
            images=[s['rgb'].to(device,non_blocking=True) for s in samples]
            depths=[s['depth'].to(device,non_blocking=True) for s in samples]
            intrinsics=torch.stack(k)
            scales=accepted.new_tensor([s.get('depth_scale',1.) for s in samples])[:,None,None,None]
        for i in range(burn+unroll):
            fs=[];roles=[]
            if batched:
                rgb=torch.stack([x[i].float()/255 if x.dtype==torch.uint8 else x[i].float() for x in images])
                depth=torch.stack([x[i].float() for x in depths])*scales
                features,diag=build_current_batch(rgb,depth,accepted,intrinsics,meshes,self.renderer,
                    times[:,i+1],times[:,i],previous,None if previous is None else times[:,i-1],
                    c['image_size'],c['crop_expansion'])
                role_bias=diag['role_bias']
            else:
                for lane,s in enumerate(samples):
                    rgb=s['rgb'][i].to(device).float();rgb=rgb/255 if s['rgb'].dtype==torch.uint8 else rgb
                    depth=s['depth'][i].to(device).float()*s.get('depth_scale',1.)
                    f,d=build_current_features(rgb,depth,accepted[lane],k[lane],meshes[lane],self.renderer,
                        times[lane,i+1],times[lane,i],None if previous is None else previous[lane],
                        None if previous is None else times[lane,i-1],c['image_size'],c['crop_expansion'])
                    fs.append(f);roles.append(d['role_bias'])
                features=stack_current(fs);role_bias=torch.stack(roles)
            meta=FrameMeta(times[:,i+1].clone(),torch.tensor([int(s['frames'][i+1]) for s in samples],device=device),
                torch.arange(b,device=device,dtype=torch.int64),torch.ones(b,17,device=device,dtype=torch.bool),role_bias)
            with torch.set_grad_enabled(torch.is_grad_enabled() and i>=burn):
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=c['precision']=='bf16'):
                    out,cache=self.tracker(features,meta,cache)
            # The current GT is accessed only after input, source and prediction exist.
            if i>=burn:
                if 'cross_attention_output' in out:
                    h=out['cross_attention_output'].detach();g=out['context_gate'].detach()
                    cross_diagnostics.append(torch.stack((g.mean(),(g*h).norm(dim=-1).mean(),out['latent_object'].detach().norm(dim=-1).mean())))
                target=torch.stack([s['targets'][i].to(device) for s in samples]).float()
                loss,detail=pose_loss(out['pose_centered'],target,points,diameter)
                losses.append(loss)
                metrics.append(torch.stack((detail['translation'],detail['rotation'],detail['points'],
                    (out['pose_centered'][:,:3,3]-target[:,:3,3]).norm(dim=-1).mean()*1000,
                    angle(out['pose_centered'][:,:3,:3]@target[:,:3,:3].transpose(-1,-2)).mean()*180/torch.pi)).detach())
            if return_predictions:predictions.append(out['pose_centered'])
            previous=accepted;accepted=out['pose_centered'].detach()
            if i==burn-1:cache=cache.detach()
        # Cache is local to this fragment and cannot survive optimizer.step.
        result=dict(loss=torch.stack(losses).mean(),metrics=torch.stack(metrics).mean(0),supervised_frames=b*unroll,
                    cache_bytes=cache.kv_bytes,kv_has_training_graph=any(block.key.grad_fn is not None for layer in cache.layers for block in layer))
        if return_predictions:result['predictions']=torch.stack(predictions,1)
        if cross_diagnostics:
            result['cross_diagnostics']=torch.stack(cross_diagnostics).mean(0)
            result['context_kv_has_training_graph']=all(b.key.grad_fn is not None and b.value.grad_fn is not None for b in cache.contexts[-1:])
        return result
