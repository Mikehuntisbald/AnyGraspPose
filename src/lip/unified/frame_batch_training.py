"""Batch independent recovery frames after an immutable causal crop pass."""
import torch
from dataclasses import fields,replace
from .features import prepare_scene,encode_scenes,build_teachers
from .timing import Timings


def independent_episode(runner,episodes,targets,backward=True):
    model=runner.model;c=runner.config;reference=runner.reference_model
    from .execution_speed import configure_execution
    configure_execution(runner)
    assert model.disable_history and reference is not None
    assert not c['runtime']['history_pair_frames'] and c['training']['history_pair_weight']==0
    assert not c['runtime']['pose_pair_frames']
    profile=Timings();model.profile_timing=profile
    device=episodes[0].rgb.device;b=len(episodes);count=len(episodes[0].times)
    gt=torch.stack([x[0] for x in targets]);visible=torch.stack([x[1] for x in targets])
    poses=torch.stack([e.initial for e in episodes]);previous=None;reference_memory=None
    history=torch.tensor([e.history for e in episodes],device=device)
    cad_enabled=torch.tensor([e.cad_enabled for e in episodes],device=device)
    diameter=poses.new_tensor([float(e.mesh['diameter']) for e in episodes])
    points=torch.stack([torch.as_tensor(e.mesh['points'],device=device) for e in episodes])
    scene_frames=[];occlusion_frames=[];runner.last_poses=[]
    trusted=getattr(model,'trusted_training_inputs',False);model.trusted_training_inputs=True
    try:
        # This prepass reproduces the original reference trajectory exactly.
        # No student output or teacher label influences its input or pose update.
        with torch.no_grad():
            for frame in range(count):
                with profile.record('crop_and_student_render'):
                    scenes=[prepare_scene(e.rgb[frame],e.depth[frame],poses[i],e.mesh,e.k,e.times[frame],e.stream,e.cad,
                        runner.renderer,None if previous is None else previous[i],None if frame==0 else e.times[frame-1],fast=c['runtime'].get('fast_geometry',False)) for i,e in enumerate(episodes)]
                with profile.record('textured_rgbd_occlusion'):
                    occlusions=[e.occlusion_plan.render(s,frame) for e,s in zip(episodes,scenes)]
                with profile.record('fixed_crop_reference'),torch.autocast(device.type,dtype=torch.bfloat16):
                    obs=encode_scenes(reference,scenes,cad_enabled=cad_enabled,frame_id=frame,occlusions=occlusions)
                    output,reference_memory=reference(obs,reference_memory,history)
                scene_frames.append(scenes);occlusion_frames.append(occlusions)
                if frame:
                    previous=poses;poses=output['pose_centered'].detach();runner.last_poses.append(poses)
        losses=[];metrics=[];batch_frames=c['runtime']['frame_batch']
        # Frame zero has no recovery loss and no student history to write.
        for start in range(1,count,batch_frames):
            ids=list(range(start,min(count,start+batch_frames)));frames=len(ids)
            scenes=[s for i in ids for s in scene_frames[i]]
            occlusions=[o for i in ids for o in occlusion_frames[i]]
            truth=torch.cat([gt[:,i] for i in ids]);masks=torch.cat([visible[:,i] for i in ids])
            obs=encode_scenes(model,scenes,cad_enabled=cad_enabled.repeat(frames),frame_id=start,occlusions=occlusions)
            with profile.record('teacher'):
                target=build_teachers(model.ema_teacher,scenes,truth,masks,[o.mask for o in occlusions],runner.renderer,
                    real_geometry_max_radius_d=c['supervision'].get('real_geometry_max_radius_d'),
                    fast=c['runtime'].get('fast_geometry',False),batch_render=c['runtime'].get('batch_teacher_render',False),
                    vectorized=c['runtime'].get('vectorized_teacher',False))
                from .cad_surface_targets import surface_targets
                target=surface_targets(model,scenes,target)
            with profile.record('jepa_and_pose'),torch.autocast(device.type,dtype=torch.bfloat16):
                output,_=model(obs,None,torch.zeros(b*frames,device=device,dtype=torch.bool))
            frame_losses=[];frame_metrics=[]
            with profile.record('loss'),torch.autocast(device.type,dtype=torch.bfloat16):
                for j,frame in enumerate(ids):
                    sl=slice(j*b,(j+1)*b)
                    part={k:(v[sl] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==b*frames else v) for k,v in output.items()}
                    teacher=replace(target,**{f.name:getattr(target,f.name)[sl] for f in fields(target) if isinstance(getattr(target,f.name),torch.Tensor)})
                    value,stats=runner.loss(part,teacher,gt[:,frame],points,diameter,c['training']['loss_weights'])
                    frame_losses.append(value);frame_metrics.append(stats)
            # Preserve each frame's valid-lane normalization before averaging.
            # A single global valid-lane mean would change the objective.
            loss=torch.stack(frame_losses).sum()/(count-1)
            if backward:
                with profile.record('backward'):loss.backward()
            losses.append(loss.detach());metrics.append(torch.stack(frame_metrics).sum(0)/(count-1))
    finally:
        model.trusted_training_inputs=trusted;model.profile_timing=None
    runner.profile=profile
    runner.history_pair_metrics=poses.new_zeros(2);runner.pose_pair_metrics=poses.new_zeros(2)
    runner.history_supported_target_fraction=poses.new_zeros(())
    return torch.stack(losses).sum(),torch.stack(metrics).sum(0)
