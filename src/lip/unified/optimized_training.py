"""Whole-episode semantics with segmented backward and one gradient collective."""
import time
from dataclasses import replace
import torch
import torch.distributed as dist
from lip.losses import pose_loss
from .features import prepare_scene,encode_scenes,build_teachers
from .losses import reconstruction_loss,RECONSTRUCTION_METRICS
from .timing import Timings


def error_target(base,truth,diameter):
    from lip.geometry.so3 import log
    return torch.cat((log(truth[:,:3,:3].float()@base[:,:3,:3].float().transpose(-1,-2)),
        (truth[:,:3,3].float()-base[:,:3,3].float())/diameter[:,None]),-1).detach()


def paired_response_loss(first,second,truth,diameter,enabled):
    a=error_target(first['pose_base'],truth,diameter);b=error_target(second['pose_base'],truth,diameter)
    target=a-b;scale=target.norm(dim=-1,keepdim=True).clamp_min(.01)
    valid=enabled&(a[:,:3].norm(dim=-1)<2.8)&(b[:,:3].norm(dim=-1)<2.8)
    total=target.new_zeros(len(target))
    for pa,pb in [(first['pose_error_prediction'],second['pose_error_prediction']),
                  (torch.cat((first['delta_rotvec'],first['delta_center_norm']),-1),
                   torch.cat((second['delta_rotvec'],second['delta_center_norm']),-1))]:
        total+=torch.nn.functional.smooth_l1_loss((pa.float()-pb.float())/scale,target/scale,beta=.1,reduction='none').mean(-1)
    return (total*valid).mean()


def combined_loss(output,teacher,truth,points,diameter,weights):
    pose,parts=pose_loss(output['pose_centered'],truth,points,diameter)
    reconstruction,rec=reconstruction_loss(output,teacher,weights)
    metrics=torch.stack([pose.detach(),parts['translation'].detach(),parts['rotation'].detach(),parts['points'].detach(),
                         *[rec[k] for k in RECONSTRUCTION_METRICS]])
    auxiliary=pose.new_zeros(())
    if 'pose_error_prediction' in output:
        from lip.geometry.so3 import log
        with torch.autocast(pose.device.type,enabled=False):
            base=output['pose_base'].float()
            target=torch.cat((log(truth[:,:3,:3].float()@base[:,:3,:3].transpose(-1,-2)),
                              (truth[:,:3,3].float()-base[:,:3,3])/diameter[:,None]),-1).detach()
            error=torch.nn.functional.smooth_l1_loss(output['pose_error_prediction'].float(),target,beta=.02,reduction='none')
            auxiliary=.5*error[:,:3].mean()+error[:,3:].mean()
        metrics=torch.cat((metrics,auxiliary.detach()[None]))
    return pose+reconstruction+weights.get('pose_error',0.)*auxiliary,metrics


class OptimizedEpisode:
    """No optimizer calls within an episode; gradients accumulate across TBPTT.

    Set segmented=False for the mathematically equivalent reference backward.
    Whole-episode normalization stays count-1, including partial first segment.
    """
    def __init__(self,model,renderer,config,reference_model=None):
        self.model=model;self.renderer=renderer;self.config=config
        self.reference_model=reference_model
        objective=combined_loss
        if 'reconstruction_only' in config:
            from .reconstruction_only import recovery_objective
            if reference_model is None or any(p.requires_grad for p in reference_model.parameters()):
                raise ValueError('Recovery-only training requires an immutable crop reference')
            objective=recovery_objective
            if config.get("joint_pose",{}).get("enabled"):
                from .joint_pose import joint_objective
                objective=joint_objective
        if config.get('pose_geometry_only',{}).get('enabled'):
            from .pose_geometry import objective,FEATURE_WEIGHTS
            assert config.get('joint_pose',{}).get('enabled') and config['runtime']['frame_batch']>1
            assert all(config['training']['loss_weights'].get(k,0)==0 for k in FEATURE_WEIGHTS)
        self.loss=torch.compile(objective,fullgraph=True,dynamic=False) if config['runtime'].get('compile_loss',False) else objective

    def __call__(self,episodes,targets,*,backward=True,segmented=True):
        if self.config["runtime"].get("frame_batch",1)>1:
            from .frame_batch_training import independent_episode
            return independent_episode(self,episodes,targets,backward=backward)
        profile=Timings();model=self.model;model.profile_timing=profile
        device=episodes[0].rgb.device
        poses=torch.stack([e.initial for e in episodes]);previous=None;memory=None;reference_memory=None
        gt=torch.stack([t[0] for t in targets]);visible=torch.stack([t[1] for t in targets])
        points=torch.stack([torch.as_tensor(e.mesh['points'],device=device) for e in episodes])
        diameter=poses.new_tensor([float(e.mesh['diameter']) for e in episodes])
        history=torch.tensor([e.history for e in episodes],device=device)
        cad_enabled=torch.tensor([e.cad_enabled for e in episodes],device=device)
        count=len(episodes[0].times);segment=self.config['training']['tbptt_frames']
        pair_frames=[i for i in self.config['runtime'].get('pose_pair_frames',[]) if i<count]
        if any(i<=0 or i>=self.config['training']['anchor_frames'] for i in pair_frames):
            raise ValueError('Paired views must use clean anchor frames')
        history_frames=[i for i in self.config['runtime'].get('history_pair_frames',[]) if i<count]
        focus = self.config.get('recovery_focus',{}).get('enabled',False)
        anchor = self.config['training']['anchor_frames']
        allowed_history = bool(history_frames) and history_frames[0]==anchor and all(i>=anchor for i in history_frames)
        if history_frames and ((not focus and history_frames!=[anchor]) or not allowed_history or anchor<=0):
            raise ValueError('History pair must cross the clean/occluded boundary')
        if history_frames and model.architecture_id not in ('stream_conv_cross_geohistory_jepa_v10','stream_conv_cross_supported_history_jepa_v10','stream_conv_cross_dense_history_jepa_v10','stream_recovered_relation_jepa_v11','stream_cad_surface_jepa_v12'):
            raise ValueError('History pair requires an explicit history-repair architecture')
        previous_observation=None;clean_anchor=None;clean_anchor_scenes=None;history_metrics=[];history_support=[]
        paired_metrics=[]
        if count<2 or any(len(e.times)!=count for e in episodes):raise ValueError('Episode length mismatch')
        losses=[];metrics=[];pending=[];self.last_poses=[]
        trusted=getattr(model,'trusted_training_inputs',False);model.trusted_training_inputs=True
        try:
            for frame in range(count):
                with profile.record('crop_and_student_render'):
                    scenes=[prepare_scene(e.rgb[frame],e.depth[frame],poses[i],e.mesh,e.k,e.times[frame],e.stream,e.cad,
                        self.renderer,None if previous is None else previous[i],None if frame==0 else e.times[frame-1]) for i,e in enumerate(episodes)]
                with profile.record('textured_rgbd_occlusion'):
                    occlusions=[e.occlusion_plan.render(s,frame) for e,s in zip(episodes,scenes)]
                obs=encode_scenes(model,scenes,cad_enabled=cad_enabled,frame_id=frame,occlusions=occlusions)
                if frame:
                    with profile.record('teacher'):
                        teacher=build_teachers(getattr(model,"ema_teacher",model.encoder),scenes,gt[:,frame],visible[:,frame],[o.mask for o in occlusions],self.renderer,
                            real_features=(obs.mid,obs.last) if self.config['runtime'].get('reuse_teacher_real',True) else None,
                            reuse_real=[not o.provenance for o in occlusions],
                            real_geometry_max_radius_d=self.config.get('supervision',{}).get('real_geometry_max_radius_d'))
                        if hasattr(model,'cad_surface'):
                            from .cad_surface_targets import surface_targets
                            teacher=surface_targets(model,scenes,teacher)
                incoming_memory=memory
                with profile.record('jepa_and_pose'),torch.autocast(device.type,dtype=torch.bfloat16):
                    output,memory=model(obs,incoming_memory,history)
                reference_output=output
                if self.reference_model is not None:
                    with profile.record('fixed_crop_reference'),torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16):
                        reference_obs = obs
                        if getattr(model, 'trainable_encoder', False):
                            reference_obs = encode_scenes(self.reference_model,scenes,cad_enabled=cad_enabled,frame_id=frame,occlusions=occlusions)
                        reference_output,reference_memory=self.reference_model(reference_obs,reference_memory,history)
                if frame:
                    with profile.record('loss'),torch.autocast(device.type,dtype=torch.bfloat16):
                        loss,parts=self.loss(output,teacher,gt[:,frame],points,diameter,self.config['training']['loss_weights'])
                    extra=loss.new_zeros(())
                    if frame in pair_frames:
                        from lip.geometry.so3 import update
                        suffix=model.weights_version.rsplit('/',1)[-1]
                        hint=int(suffix) if suffix.isdigit() else 0
                        axes=(torch.arange(len(episodes),device=device)+frame+hint)%6
                        perturb=poses.new_zeros(len(episodes),6)
                        perturb[torch.arange(len(episodes),device=device),axes]=torch.where(axes<3,torch.pi/18,.05)*(1 if frame==pair_frames[0] else -1)
                        cf_base=update(poses,perturb[:,:3],perturb[:,3:],diameter)
                        with profile.record('crop_and_student_render'):
                            cf_scenes=[prepare_scene(e.rgb[frame],e.depth[frame],cf_base[i],e.mesh,e.k,e.times[frame],e.stream,e.cad,
                                self.renderer,None if previous is None else previous[i],e.times[frame-1]) for i,e in enumerate(episodes)]
                        cf_obs=encode_scenes(model,cf_scenes,cad_enabled=cad_enabled,frame_id=frame)
                        # Do not reveal the synthetic perturbation through a motion jump.
                        state=cf_obs.state.clone();state[:,14:23]=obs.state[:,14:23]
                        cf_obs=replace(cf_obs,state=state)
                        with profile.record('pose_pair_forward'),torch.autocast(device.type,dtype=torch.bfloat16):
                            cf_output,_discarded_memory=model(cf_obs,incoming_memory,history)
                        with profile.record('pose_pair_loss'):
                            pair=paired_response_loss(output,cf_output,gt[:,frame],diameter,cad_enabled)
                            absolute,_=pose_loss(cf_output['pose_centered'],gt[:,frame],points,diameter,valid=cad_enabled)
                            extra=self.config['training'].get('pose_pair_weight',.5)*pair+self.config['training'].get('pose_pair_absolute_weight',.25)*absolute
                        paired_metrics.append(torch.stack((pair.detach(),absolute.detach())))
                    pair_history=loss.new_zeros(())
                    if frame in history_frames:
                        from .history_training import history_pair_loss,observed_history_support
                        with profile.record('history_pair_writer_and_recovery'),torch.autocast(device.type,dtype=torch.bfloat16):
                            support=None
                            if focus:
                                support=observed_history_support(clean_anchor_scenes,gt[:,anchor-1],visible[:,anchor-1],teacher)
                                history_support.append(((support&teacher.geometry_real_weight).float().sum()/teacher.geometry_real_weight.sum().clamp_min(1)).detach())
                            pair_history,history_parts=history_pair_loss(model,clean_anchor if focus else previous_observation,obs,teacher,
                                self.config['training']['loss_weights'] if focus else None,support)
                        pair_history=pair_history*self.config['training']['history_pair_weight']/len(history_frames)
                        history_metrics.append(history_parts)
                    pending.append(loss/(count-1)+extra/max(1,len(pair_frames))+pair_history)
                    losses.append((loss+extra*(count-1)/max(1,len(pair_frames))+pair_history*(count-1)).detach());metrics.append(parts.detach())
                    previous=poses;poses=reference_output['pose_centered'].detach()
                    self.last_poses.append(poses)
                previous_observation=obs
                if frame==anchor-1:clean_anchor=obs;clean_anchor_scenes=scenes
                if (frame+1)%segment==0 or frame==count-1:
                    memory=memory.detach()
                    if backward and segmented and pending:
                        with profile.record('backward'):torch.stack(pending).sum().backward()
                        pending.clear()
                    if getattr(model, "trainable_encoder", False):
                        from .ema_encoder import detach_observation
                        if clean_anchor is not None: clean_anchor=detach_observation(clean_anchor)
                        if previous_observation is not None: previous_observation=detach_observation(previous_observation)
            if backward and not segmented:
                with profile.record('backward'):torch.stack(pending).sum().backward()
        finally:
            model.trusted_training_inputs=trusted;model.profile_timing=None
        self.history_pair_metrics=torch.stack(history_metrics).mean(0) if history_metrics else poses.new_zeros(2)
        self.history_supported_target_fraction=torch.stack(history_support).mean() if history_support else poses.new_zeros(())
        self.profile=profile
        self.pose_pair_metrics=torch.stack(paired_metrics).mean(0) if paired_metrics else poses.new_zeros(2)
        return torch.stack(losses).mean(),torch.stack(metrics).mean(0)


def synchronize_gradients(parameters):
    """All ranks participate once, including locally unused parameter rows.

    Frozen parameters are omitted. Fixed ordering handles differing history /
    CAD masks across ranks without dynamic unused-parameter graph traversal.
    """
    params=[p for p in parameters if p.requires_grad]
    if not params:return 0
    gradients=[(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1) for p in params]
    flags=params[0].new_tensor([p.grad is not None for p in params])
    flat=torch.cat([*gradients,flags])
    if dist.is_initialized():
        dist.all_reduce(flat)
    used=flat[-len(params):].bool().tolist()
    if dist.is_initialized():flat[:-len(params)].div_(dist.get_world_size())
    offset=0
    for p,present in zip(params,used):
        p.grad=flat[offset:offset+p.numel()].view_as(p) if present else None
        offset+=p.numel()
    return flat.numel()*flat.element_size()


def make_optimizer(model,config,*,fused=True):
    from .model import parameter_category
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            from .pose_geometry import inactive_feature_parameter
            if not (config.get('pose_geometry_only',{}).get('enabled') and inactive_feature_parameter(name)):continue
        category=parameter_category(name)
        if getattr(model, "trainable_encoder", False) and name.startswith("encoder."):
            category="encoder"
        if category not in (None, 'encoder') and model.architecture_id in ('stream_recovered_relation_jepa_v11','stream_cad_surface_jepa_v12'):
            category='new' if name.startswith(('geometry_readout.','cad_surface.')) else ('pose' if name.startswith('head.') else 'predictor')
        if getattr(model,'surface_decoder_kind','mlp')=='dpt' and name.startswith('surface_head.'):
            category='new'
        if config.get("joint_pose",{}).get("enabled"):
            from .reconstruction_only import is_pose_parameter
            if is_pose_parameter(name):category="pose"
        if category is not None:
            groups.setdefault((category,p.ndim>1),[]).append((name,p))
    t=config['training']
    return torch.optim.AdamW([dict(params=[p for _,p in values],names=[n for n,_ in values],category=category,
        lr=t['learning_rates'][category],weight_decay=t['weight_decay'] if decay else 0.) for (category,decay),values in sorted(groups.items())],
        betas=(.9,.95),fused=fused)
