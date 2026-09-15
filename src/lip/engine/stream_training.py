"""A whole burn-in/unroll is one DDP forward; pose feedback detaches, KV does not."""
import torch
from torch import nn
from lip.engine.stream_features import build_current_features,stack_current,mesh_to_device
from lip.engine.stream_state import FrameMeta
from lip.data.perturb import noisy_history
from lip.losses import pose_loss
from lip.geometry.so3 import angle,center_pose


def supervision_positions(burn,unroll,startup_frames=0):
    """Keep the same observed fragment and target count when training startup.

    Enabling startup labels replaces evenly spaced later labels, rather than
    adding targets or changing augmentation timing. Visual KV is still detached
    at the original burn boundary; pose feedback always detaches.
    """
    if burn<0 or unroll<1 or startup_frames not in (0,8):raise ValueError('Unsupported supervision window')
    if startup_frames and (burn!=8 or unroll!=48):raise ValueError('Startup supervision v1 requires the fixed 8+48 fragment')
    selected=[False]*burn+[True]*unroll
    if startup_frames:
        selected[:startup_frames]=[True]*startup_frames
        for relative in range(unroll//startup_frames-1,unroll,unroll//startup_frames):selected[burn+relative]=False
    assert sum(selected)==unroll
    return tuple(selected)


class StreamTrainingModule(nn.Module):
    def __init__(self,tracker,config,renderer):
        super().__init__();self.tracker=tracker;self.config=config;self.renderer=renderer

    def forward(self,samples,return_predictions=False):
        c=self.config;device=next(self.tracker.parameters()).device;burn=c['burn_in_frames'];unroll=c['supervised_unroll_frames']
        supervised=supervision_positions(burn,unroll,c.get('startup_supervision_frames',0))
        b=len(samples);meshes=[mesh_to_device(s['mesh'],device) for s in samples]
        accepted=torch.stack([s['initial_pose'].to(device).float() for s in samples]);previous=None
        if c['initial_pose_noise']:
            accepted=torch.stack([noisy_history(p[None],meshes[i]['diameter'],torch.Generator().manual_seed(samples[i]['sample']['seed']))[0][0] for i,p in enumerate(accepted)])
        # Both arms generate identical synthetic occluders from this common
        # training-only noisy prior, independent of real-initializer assignment.
        augmentation_prior=accepted
        prime=c.get('prime_initial_observation',False)
        if c.get('real_initialization_probability',0.) and any('real_initialization_requested' not in s for s in samples):
            raise ValueError('Real initialization requested without dataset provenance')
        real=[s.get('real_initial_pose') if s.get('real_initialization_requested',False) else None for s in samples]
        if any(p is not None for p in real):
            if not prime:raise ValueError('Real initializers require an encoded initial observation')
            accepted=torch.stack([center_pose(samples[i]['real_initial_pose_original'].to(device).float(),meshes[i]['center'].float())
                if p is not None and 'real_initial_pose_original' in samples[i] else p.to(device).float() if p is not None else accepted[i]
                for i,p in enumerate(real)])
        times=torch.stack([s['timestamps'].to(device,dtype=torch.float64) for s in samples])
        k=[s['k'].to(device).float() for s in samples];cache=None;losses=[];metrics=[];predictions=[];cross_diagnostics=[];rk_diagnostics=[];spatial_diagnostics=[];aligned_diagnostics=[]
        diameter=torch.stack([m['diameter'].float() for m in meshes]);points=torch.stack([m['points'].float() for m in meshes])
        batched=c.get('batch_current_features',False)
        plans=None;occlusion_diagnostics=[];direct_pose_diagnostics=[];reference_diagnostics=[];reference_write_diagnostics=[];rotation_anchor_diagnostics=[];smooth_rotation_diagnostics=[];spatial_alignment_diagnostics=[]
        if c.get('temporal_occlusion_probability',0.):
            from lip.data.temporal_occlusion import make_plan,composite
            plans=[]
            for lane,s in enumerate(samples):
                first=(s['initial_rgb'] if prime else s['rgb'][0]).to(device).float()
                if s['rgb'].dtype==torch.uint8:first=first/255
                plans.append(make_plan(first,augmentation_prior[lane],k[lane],meshes[lane]['vertices'],meshes[lane]['diameter'],
                    s['sample']['seed'],burn+unroll+int(prime),burn+int(prime),c['temporal_occlusion_probability'],c.get('temporal_occlusion_startup_probability',0.)))
        if batched:
            from lip.engine.stream_batch_features import build_current_batch
            # Upload each raw fragment once. Images remain uint8/uint16 until
            # their current frame is consumed; future frames never enter features.
            images=[s['rgb'].to(device,non_blocking=True) for s in samples]
            depths=[s['depth'].to(device,non_blocking=True) for s in samples]
            intrinsics=torch.stack(k)
            scales=accepted.new_tensor([s.get('depth_scale',1.) for s in samples])[:,None,None,None]
        if prime:
            if self.tracker.architecture_id!='stream_dual_cross_residual':raise ValueError('Primed training requires the retained residual tracker')
            warm_rgb=torch.stack([s['initial_rgb'].to(device).float()/255 if s['initial_rgb'].dtype==torch.uint8 else s['initial_rgb'].to(device).float() for s in samples])
            warm_depth=torch.stack([s['initial_depth'].to(device).float()*s.get('depth_scale',1.) for s in samples])
            if plans is not None:
                augmented=[composite(warm_rgb[j],warm_depth[j],plans[j],0) for j in range(b)]
                warm_rgb=torch.stack([v[0] for v in augmented]);warm_depth=torch.stack([v[1] for v in augmented])
            warm_prev=times[:,0]-times.new_tensor([s.get('nominal_frame_interval',c['time_unit']) for s in samples])
            if batched:
                warm_features,warm_diag=build_current_batch(warm_rgb,warm_depth,accepted,intrinsics,meshes,self.renderer,
                    times[:,0],warm_prev,None,None,c['image_size'],c['crop_expansion'])
                warm_bias=warm_diag['role_bias']
            else:
                fs=[];biases=[]
                for j in range(b):
                    f,d=build_current_features(warm_rgb[j],warm_depth[j],accepted[j],k[j],meshes[j],self.renderer,
                        times[j,0],warm_prev[j],None,None,c['image_size'],c['crop_expansion'])
                    fs.append(f);biases.append(d['role_bias'])
                warm_features=stack_current(fs);warm_bias=torch.stack(biases)
            warm_meta=FrameMeta(times[:,0].clone(),torch.ones(b,device=device,dtype=torch.int64),
                torch.arange(b,device=device,dtype=torch.int64),torch.ones(b,17,device=device,dtype=torch.bool),warm_bias)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=c['precision']=='bf16'):
                _,cache=self.tracker(warm_features,warm_meta,None)
            # Retain the supplied pose and previous=None, as online priming does.
            # The current frame's feature cache can receive subsequent training credit.
        for i in range(burn+unroll):
            fs=[];roles=[];occlusion_masks=[]
            if batched:
                rgb=torch.stack([x[i].float()/255 if x.dtype==torch.uint8 else x[i].float() for x in images])
                depth=torch.stack([x[i].float() for x in depths])*scales
                if plans is not None:
                    augmented=[composite(rgb[j],depth[j],plans[j],i+int(prime)) for j in range(b)]
                    rgb=torch.stack([v[0] for v in augmented]);depth=torch.stack([v[1] for v in augmented])
                    occlusion_masks=[v[2] for v in augmented]
                features,diag=build_current_batch(rgb,depth,accepted,intrinsics,meshes,self.renderer,
                    times[:,i+1],times[:,i],previous,None if previous is None else times[:,i-1],
                    c['image_size'],c['crop_expansion'])
                role_bias=diag['role_bias']
            else:
                for lane,s in enumerate(samples):
                    rgb=s['rgb'][i].to(device).float();rgb=rgb/255 if s['rgb'].dtype==torch.uint8 else rgb
                    depth=s['depth'][i].to(device).float()*s.get('depth_scale',1.)
                    if plans is not None:
                        rgb,depth,mask=composite(rgb,depth,plans[lane],i+int(prime));occlusion_masks.append(mask)
                    f,d=build_current_features(rgb,depth,accepted[lane],k[lane],meshes[lane],self.renderer,
                        times[lane,i+1],times[lane,i],None if previous is None else previous[lane],
                        None if previous is None else times[lane,i-1],c['image_size'],c['crop_expansion'])
                    fs.append(f);roles.append(d['role_bias'])
                features=stack_current(fs);role_bias=torch.stack(roles)
            meta=FrameMeta(times[:,i+1].clone(),torch.tensor([i+2 if prime else int(s['frames'][i+1]) for s in samples],device=device),
                torch.arange(b,device=device,dtype=torch.int64),torch.ones(b,17,device=device,dtype=torch.bool),role_bias)
            with torch.set_grad_enabled(torch.is_grad_enabled() and (i>=burn or supervised[i])):
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=c['precision']=='bf16'):
                    out,cache=self.tracker(features,meta,cache)
            # The current GT is accessed only after input, source and prediction exist.
            if supervised[i]:
                if 'rotation_anchor_coefficient' in out:
                    value=out['rotation_anchor_coefficient'].detach().float()
                    smooth_rotation_diagnostics.append(torch.stack((value.mean(),value.abs().mean(),
                        (value>0).float().mean(),(value<0).float().mean(),out['rotation_anchor_gap_norm'].detach().float().mean())))
                if 'rotation_anchor_fraction' in out:
                    rotation_anchor_diagnostics.append(torch.stack((out['rotation_anchor_fraction'].detach().float().mean(),out['rotation_anchor_gap_norm'].detach().float().mean())))
                if 'reference_write_rotation_coefficient' in out:
                    names=('reference_write_rotation_coefficient','reference_write_center_coefficient','reference_write_rotation_norm','reference_write_center_norm','reference_write_valid_target')
                    reference_write_diagnostics.append(torch.stack([out[name].detach().float().mean() for name in names]))
                if 'reference_rotation_coefficient' in out:
                    names=('reference_rotation_coefficient','reference_center_coefficient','reference_rotation_residual_norm','reference_center_residual_norm','reference_age_frames')
                    reference_diagnostics.append(torch.stack([out[name].detach().mean() for name in names]))
                if 'direct_rotation_norm' in out:
                    direct_pose_diagnostics.append(torch.stack((out['direct_rotation_norm'].detach().mean(),out['direct_center_norm'].detach().mean())))
                if 'aligned_update_norm' in out:
                    aligned_diagnostics.append(torch.stack((out['aligned_update_norm'].detach().mean(),out['aligned_tokens_read'].float().mean(),out['aligned_queries'].float().mean())))
                if occlusion_masks:
                    masks=torch.stack(occlusion_masks)
                    occlusion_diagnostics.append(torch.stack((masks.float().mean(),masks.flatten(1).any(1).float().mean())))
                if 'spatial_update_norm' in out:
                    spatial_diagnostics.append(torch.stack((out['spatial_update_norm'].detach().mean(),out['spatial_tokens_read'].float().mean())))
                if 'observation_support' in out:
                    rk_diagnostics.append(torch.stack((out['observation_support'].detach().mean(),out['anchors_read'].float().mean())))
                if 'cross_attention_output' in out:
                    h=out['cross_attention_output'].detach();g=out['context_gate'].detach()
                    cross_diagnostics.append(torch.stack((g.mean(),(g*h).norm(dim=-1).mean(),out['latent_object'].detach().norm(dim=-1).mean())))
                target=torch.stack([s['targets'][i].to(device) for s in samples]).float()
                loss,detail=pose_loss(out['pose_centered'],target,points,diameter)
                if c.get('alignment_spatial_weight',0.) and self.tracker.training:
                    from lip.losses_spatial_alignment import correspondence_targets,spatial_alignment_loss
                    eligible=torch.tensor([int(s['stream']['object_id']) not in c['alignment_symmetric_object_ids'] for s in samples],device=device,dtype=torch.bool)
                    gt_depth=torch.stack([self.renderer(m,target[j].detach(),diag['K_crop'][j],c['image_size'])[0] if bool(eligible[j]) else depth.new_zeros(1,224,224) for j,m in enumerate(meshes)])
                    distribution,supported,coverage=correspondence_targets(features['geometry'],diameter,target,diag['K_crop'],diag['A'],depth,gt_depth,eligible)
                    logprob=self.tracker.rotation_alignment.correspondence_log_probability(*out['alignment_matching_inputs'])
                    auxiliary,accuracy=spatial_alignment_loss(logprob,distribution,supported)
                    loss=loss+c['alignment_spatial_weight']*auxiliary
                    spatial_alignment_diagnostics.append(torch.stack((auxiliary.detach(),accuracy.detach(),coverage['visible_keys'].float(),coverage['supported_queries'].float(),coverage['eligible_clips'].float())))
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
        assert len(losses)==unroll
        result['startup_supervised_frames']=b*sum(supervised[:burn])
        result['omitted_late_targets']=b*sum(not flag for flag in supervised[burn:])
        if prime:
            result.update(real_initializations=sum(p is not None for p in real),
                real_initializations_requested=sum(s.get('real_initialization_requested',False) for s in samples),
                real_initializations_missing=sum(s.get('real_initialization_missing',False) for s in samples),primed_observations=b)
        if return_predictions:result['predictions']=torch.stack(predictions,1)
        if spatial_alignment_diagnostics:result['spatial_alignment_diagnostics']=torch.stack(spatial_alignment_diagnostics).mean(0)
        if cross_diagnostics:
            result['cross_diagnostics']=torch.stack(cross_diagnostics).mean(0)
            result['context_kv_has_training_graph']=all(b.key.grad_fn is not None and b.value.grad_fn is not None for b in cache.contexts[-1:])
        if rk_diagnostics:result['rk_diagnostics']=torch.stack(rk_diagnostics).mean(0)
        if spatial_diagnostics:result['spatial_diagnostics']=torch.stack(spatial_diagnostics).mean(0)
        if aligned_diagnostics:result['aligned_diagnostics']=torch.stack(aligned_diagnostics).mean(0)
        if occlusion_diagnostics:result['occlusion_diagnostics']=torch.stack(occlusion_diagnostics).mean(0)
        if direct_pose_diagnostics:result['direct_pose_diagnostics']=torch.stack(direct_pose_diagnostics).mean(0)
        if reference_diagnostics:result['reference_diagnostics']=torch.stack(reference_diagnostics).mean(0)
        if reference_write_diagnostics:
            result['reference_write_diagnostics']=torch.stack(reference_write_diagnostics).mean(0)
            result['reference_has_training_graph']=cache.reference.pose.grad_fn is not None
        if rotation_anchor_diagnostics:
            result['rotation_anchor_diagnostics']=torch.stack(rotation_anchor_diagnostics).mean(0)
        if smooth_rotation_diagnostics:
            result['smooth_rotation_diagnostics']=torch.stack(smooth_rotation_diagnostics).mean(0)
        return result
