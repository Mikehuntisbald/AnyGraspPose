"""Bounded V21 training: paired estimates, exact-zero examples, student feedback.

Independent RNG per rank, one packed gradient collective, complete resumable
checkpoints. Main JEPA forwards use predicted completion only. The optional,
explicit oracle rehearsal trains pose-readout parameters on detached targets.
"""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
from dataclasses import replace
import numpy as np,torch,torch.distributed as dist,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_serial_completion import initialize,heldout
from lip.unified.build import make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.unified.cad_surface_targets import surface_targets
from lip.unified.serial_objective import objective,delta_target
from lip.unified.reconstruction_only import is_pose_parameter
from lip.unified.optimized_training import synchronize_gradients
from lip.unified.ema_encoder import update_ema
from lip.unified.checkpoint import save,resume,atomic_json
from lip.engine.jepa_checkpoint import sha
from lip.engine.config import check_data_gate
from lip.geometry.so3 import update


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--oracle',required=True)
    p.add_argument('--stop-at',type=int);p.add_argument('--resume');p.add_argument('--adapt-from');p.add_argument('--fidelity-from');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if sum(bool(x) for x in (a.resume,a.adapt_from,a.fidelity_from))>1:raise ValueError('Choose one continuation mechanism')
    c=yaml.safe_load(Path(a.config).read_text());rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if a.preflight:c['runtime']['formal']=False
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(c['seed']);random.seed(c['seed']+rank);np.random.seed(c['seed']+rank)
    torch.use_deterministic_algorithms(c['runtime'].get('deterministic_algorithms',True))
    torch.utils.deterministic.fill_uninitialized_memory=False
    if world>1:dist.init_process_group('nccl')
    if not a.preflight and (world!=c['runtime']['world'] or world*c['runtime']['microbatch']!=c['training']['effective_batch']):
        raise ValueError('Formal effective batch/world mismatch')
    gate=json.loads((Path(a.oracle)/'receipt.json').read_text())
    if not gate['passed'] or sha(Path(a.oracle)/'readout.pt')!=gate['checkpoint_sha256']:raise ValueError('Oracle readout gate not passed')
    model,source,new=initialize(c);del source
    oracle=torch.load(Path(a.oracle)/'readout.pt',map_location='cpu',weights_only=False)
    states=model.state_dict()
    for key,value in oracle['model'].items():
        if not is_pose_parameter(key) or key not in states or states[key].shape!=value.shape:raise ValueError('Readout migration mismatch: '+key)
        states[key].copy_(value)
    del states,oracle
    for name,parameter in model.named_parameters():
        active=not (name.startswith(('ema_teacher.','writer.','memory_position.','core.memory_')) or '.history.' in name or name.startswith('core.log_error.'))
        parameter.requires_grad_(active)
    groups={}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:continue
        kind='pose' if is_pose_parameter(name) else ('encoder' if name.startswith('encoder.') else ('new' if name.startswith(('surface_head.','cad_surface.','core.feature_')) else 'predictor'))
        groups.setdefault(kind,dict(params=[],names=[],lr=c['training']['learning_rates'][kind],category=kind))
        groups[kind]['params'].append(parameter);groups[kind]['names'].append(name)
    opt=torch.optim.AdamW(list(groups.values()),weight_decay=.01,fused=True)
    maximum=c['serial_completion']['joint_steps'];stop=a.stop_at or maximum
    if not 0<stop<=maximum:raise ValueError('Stop exceeds bounded budget')
    def rate(step):
        if 'geometry_priority' in c:
            origin=c['geometry_priority']['source_step'];progress=max(0,step-origin)
            return (.3+.7*min(1.,progress/50))*(.5+.5*.5*(1+math.cos(math.pi*min(progress,maximum-origin)/(maximum-origin))))
        return min(1.,(step+1)/50)*(.3+.7*.5*(1+math.cos(math.pi*min(step,maximum)/maximum)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,rate)
    factory=Factory(c,model,make_store(c,model));audit=check_data_gate(c['paths']['index_root'])
    source_files={str(f.relative_to(Path(__file__).resolve().parents[1])):sha(f) for d in ('src','tools','configs') for f in (Path(__file__).resolve().parents[1]/d).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    provenance=dict(split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],source_sha256=hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest(),
        initializers_sha256=factory.initializers_sha256,weights=c['weights'],source_checkpoint_sha256=c['serial_completion']['source_sha256'],
        oracle_readout_sha256=gate['checkpoint_sha256'],training_split='train',official_test_access=False,teacher_input=False,
        crop_reference='student own detached feedback; GT noisy/zero pose curricula are training-only',paired_estimates=True)
    out=Path(c['paths']['output']);out=out.parent.parent/'preflight_run' if a.preflight else out
    if rank==0:out.mkdir(parents=True,exist_ok=bool(a.resume or a.adapt_from));atomic_json(out/'provenance.json',provenance)
    if world>1:dist.barrier()
    start=0
    if a.fidelity_from:
        plan=c['geometry_priority']
        if sha(a.fidelity_from)!=plan['source_sha256']:raise ValueError('Fidelity source hash mismatch')
        previous=torch.load(a.fidelity_from,map_location='cpu',weights_only=False)
        if previous['step']!=plan['source_step']:raise ValueError('Fidelity source step mismatch')
        old=previous['provenance'];parent_config=previous['config'];del previous
        for key in ('weights','seed','architecture_id','supervision','surface_decoder','dino_layers','cad_surface','staged_rope'):
            if c[key]!=parent_config[key]:raise ValueError('Fidelity changed non-objective identity: '+key)
        for key in ('split_hash','mesh_hash','initializers_sha256','weights'):
            if old[key]!=provenance[key]:raise ValueError('Fidelity data identity changed')
        if c['training']['loss_weights']!=parent_config['training']['loss_weights'] or c['training']['learning_rates']!=parent_config['training']['learning_rates']:
            raise ValueError('Paired fidelity test must retain identical scalar losses and peak rates')
        start=resume(a.fidelity_from,model,opt,scheduler,parent_config,old,rank,world)['step']
        # Adam parameters/moments and RNG were restored exactly above. Only the
        # declared continuation schedule starts a new 1000-update horizon.
        scheduler.base_lrs=[g['initial_lr'] for g in opt.param_groups]
        scheduler.last_epoch=start;scheduler._step_count=1
        for group,base_lr in zip(opt.param_groups,scheduler.base_lrs):group['lr']=base_lr*rate(start)
        scheduler._last_lr=[g['lr'] for g in opt.param_groups]
        if rank==0:atomic_json(out/'fidelity_start.json',dict(source_step=start,source_sha256=plan['source_sha256'],model_adam_rng_exact=True,
            schedule_reset=True,boundary_factor=rate(start),gradient_budget_enabled=plan['enabled'],maximum_secondary_ratio=plan['ratio']))
        save(out/'stage_start.pt',model,opt,scheduler,start,c,provenance)
    elif a.adapt_from:
        previous=torch.load(a.adapt_from,map_location='cpu',weights_only=False)
        old=previous['provenance'];previous_config=previous['config'];del previous
        if {k:v for k,v in old.items() if k!='source_sha256'}!={k:v for k,v in provenance.items() if k!='source_sha256'}:
            raise ValueError('Code-only adaptation changed data or initialization identity')
        from copy import deepcopy
        stripped=deepcopy(c);stripped['serial_completion'].pop('oracle_rehearsal_weight',None)
        old_stripped=deepcopy(previous_config);old_stripped['serial_completion'].pop('oracle_rehearsal_weight',None)
        if stripped!=old_stripped:raise ValueError('Undeclared config adaptation')
        start=resume(a.adapt_from,model,opt,scheduler,previous_config,old,rank,world)['step']
        if rank==0:atomic_json(out/f'code_adaptation{start}.json',dict(step=start,checkpoint_sha256=sha(a.adapt_from),previous_source_sha256=old['source_sha256'],source_sha256=provenance['source_sha256'],model_optimizer_scheduler_rng_exact=True,reason='Preserve readout oracle response; exact visible correspondence mask; paired observations remain identical'))
    elif a.resume:start=resume(a.resume,model,opt,scheduler,c,provenance,rank,world)['step']
    else:save(out/'initial.pt',model,opt,scheduler,0,c,provenance)
    batch=1 if a.preflight else c['runtime']['microbatch']
    # Keep rank model initialization equal; rank RNG differs only afterwards.
    if not (a.resume or a.adapt_from or a.fidelity_from):torch.manual_seed(c['seed']+rank)
    rehearsal_weight=c['serial_completion'].get('oracle_rehearsal_weight',0.)
    oracle_pack=oracle_truth=oracle_diameter=None
    if rehearsal_weight:
        cache=Path(a.oracle).parent/'oracle_cache'/f'rank{rank}'/'packets.pt'
        records=[x for x in torch.load(cache,weights_only=False) if x['split']=='train']
        oracle_pack={k:torch.cat([x['oracle'][k] for x in records]).cuda() for k in records[0]['oracle']}
        oracle_pack['feature']=torch.cat([x['predicted']['feature'] for x in records]).cuda()
        oracle_pack['weight']=oracle_pack['weight']*.5
        oracle_pack['completed_weight']=oracle_pack['completed_weight']*.5
        oracle_truth=torch.stack([x['truth'] for x in records]).cuda();oracle_diameter=oracle_truth.new_tensor([x['diameter'] for x in records])
        del records
    def episodes(step):
        result=[]
        for lane in range(batch):
            seed=9000000+(step*world+rank)*batch+lane
            for attempt in range(100):
                sample=factory.sample(seed+attempt*100000003,frames=12)
                if not heldout(sample[0].stream):result.append(sample);break
            else:raise RuntimeError('Cannot draw training sequence outside gate holdout')
        return result
    def forward(episodes,targets,bases,frame,paired_source=None,previous_bases=None):
        if paired_source is None:
            scenes=[prepare_scene(e.rgb[frame],e.depth[frame],bases[i],e.mesh,e.k,e.times[frame],e.stream+f'/lane{i}',e.cad,factory.renderer,
                None if previous_bases is None else previous_bases[i],None if previous_bases is None else e.times[frame-1],fast=True) for i,e in enumerate(episodes)]
            occ=[e.occlusion_plan.render(s,frame) for e,s in zip(episodes,scenes)]
        else:
            source_scenes,occ=paired_source;scenes=[]
            for i,s in enumerate(source_scenes):
                state=s.state.clone();state[:6]=bases[i,:3,:2].T.flatten();state[6:9]=bases[i,:3,3]/s.diameter
                scenes.append(replace(s,pose=bases[i],state=state,render=factory.renderer(s.cad['appearance'],bases[i],s.k_crop,224)))
        obs=encode_scenes(model,scenes,occlusions=occ,frame_id=frame)
        truth=torch.stack([t[0][frame] for t in targets]);mask=torch.stack([t[1][frame] for t in targets])
        teacher=build_teachers(model.ema_teacher,scenes,truth,mask,[o.mask for o in occ],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True)
        teacher=surface_targets(model,scenes,teacher)
        from lip.unified.execution_speed import crop_images_fast
        visible_measurement=torch.cat([(crop_images_fast(mask[i:i+1].float(),s.affine,mode='nearest')>.5)&~o.mask&s.bounds for i,(s,o) in enumerate(zip(scenes,occ))])
        with torch.autocast('cuda',dtype=torch.bfloat16):output,_=model(obs)
        points=torch.stack([torch.as_tensor(e.mesh['points'],device='cuda') for e in episodes])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if 'geometry_priority' in c:
                from lip.unified.serial_objective import objective_components
                loss,metrics=objective_components(output,teacher,truth,points,obs.diameter,bases,obs.measured_depth_m,c['training']['loss_weights'],visible_measurement)
            else:loss,metrics=objective(output,teacher,truth,points,obs.diameter,bases,obs.measured_depth_m,c['training']['loss_weights'],visible_measurement)
        return output,loss,metrics,obs,teacher,(scenes,occ)
    path_norms=None
    with (out/f'rank{rank}.jsonl').open('a') as log:
        for step in range(start,stop):
            begun=time.monotonic();pairs=episodes(step);ep,targets=zip(*pairs)
            frame=8 if step%2==0 else 0
            truth=torch.stack([t[0][frame] for t in targets]);diam=truth.new_tensor([float(e.mesh['diameter']) for e in ep])
            noise=torch.randn(batch,6,device='cuda')*truth.new_tensor([.15,.15,.15,.035,.035,.035])
            bases=update(truth,noise[:,:3],noise[:,3:],diam)
            # Native, potentially large errors remain represented; never run an
            # immutable reference trajectory or inspect GT during deployment.
            if step%4 in (2,3):bases=torch.stack([e.initial for e in ep])
            opt.zero_grad(set_to_none=True)
            first,l1,m1,obs,target,pair_source=forward(ep,targets,bases,frame)
            if a.preflight:
                from lip.losses import pose_loss
                pure_pose,_=pose_loss(first['pose_centered'],truth,torch.stack([torch.as_tensor(e.mesh['points'],device='cuda') for e in ep]),diam)
                gradients=torch.autograd.grad(pure_pose,[first['surface_xyz'],first['surface_depth_residual'],first['f_mid_predicted'],first['f_predicted'],first['patch_latent']],retain_graph=True)
                path_norms=[float(x.norm()) for x in gradients]
                if not all(x>0 and math.isfinite(x) for x in path_norms):raise AssertionError('Pose did not consume every completion output')
            feedback=first['pose_centered'].detach()
            # Same RGB-D observation, alternate estimated pose. Every fourth
            # example is an exact-zero base; its target is always a zero update.
            alternative=truth if step%4==0 else update(bases,-noise[:,:3],-noise[:,3:],diam)
            second,l2,m2,paired_obs,_,_=forward(ep,targets,alternative,frame,paired_source=pair_source)
            if not torch.equal(obs.packet.rgb_crop,paired_obs.packet.rgb_crop) or not torch.equal(obs.measured_depth_m,paired_obs.measured_depth_m):
                raise AssertionError('Paired estimates changed the RGB-D observation')
            pred1=torch.cat((first['delta_rotvec'],first['delta_center_norm']),-1)
            pred2=torch.cat((second['delta_rotvec'],second['delta_center_norm']),-1)
            expected=delta_target(bases,truth,diam)-delta_target(alternative,truth,diam)
            scale=truth.new_tensor([.174533]*3+[.05]*3)
            pair=F.smooth_l1_loss((pred1-pred2)/scale,expected/scale,beta=.1)
            balance=[]
            if 'geometry_priority' in c:
                geometry=(l1['geometry']+l2['geometry'])/3
                secondary=(l1['pose']+l2['pose']+l1['appearance']+l2['appearance'])/3+.1*pair
                if c['geometry_priority']['enabled']:
                    from lip.unified.geometry_priority import backward_primary_geometry
                    balance.append(backward_primary_geometry(geometry,secondary,[first['patch_latent'],second['patch_latent']],model.named_parameters(),c['geometry_priority']['ratio']))
                else:(geometry+secondary).backward()
                del geometry,secondary
            else:
                combined=(l1+l2)/3+.1*pair;combined.backward();del combined
            del first,second,obs,paired_obs,target,pair_source,l1,l2,pred1,pred2
            third,l3,m3,_,_,_=forward(ep,targets,feedback,frame+1,previous_bases=bases if 'geometry_priority' in c else None)
            if 'geometry_priority' in c:
                geometry=l3['geometry']/3;secondary=(l3['appearance']+l3['pose'])/3
                if c['geometry_priority']['enabled']:
                    balance.append(backward_primary_geometry(geometry,secondary,[third['patch_latent']],model.named_parameters(),c['geometry_priority']['ratio']))
                else:(geometry+secondary).backward()
                loss_value=float(sum(l3.values()).detach());del geometry,secondary
            else:(l3/3).backward();loss_value=float(l3.detach())
            del third,l3
            rehearsal=truth.new_zeros(())
            if rehearsal_weight:
                from lip.unified.serial_objective import oracle_readout_loss
                ids=torch.randint(len(oracle_truth),(4,),device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):rehearsal=oracle_readout_loss(model,{k:v[ids] for k,v in oracle_pack.items()},oracle_truth[ids],oracle_diameter[ids])
                (rehearsal_weight*rehearsal).backward()
            communication=time.monotonic();synchronize_gradients(model.parameters());communication=time.monotonic()-communication
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None],1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient; do not advance')
            opt.step();scheduler.step();update_ema(model)
            torch.cuda.synchronize();metrics={k:float((m1[k]+m2[k]+m3[k])/3) for k in m1}
            row=dict(step=step+1,source_step=c.get('geometry_priority',{}).get('source_step',45400),seconds=time.monotonic()-begun,loss_feedback=loss_value,pair=float(pair.detach()),
                metrics=metrics,oracle_rehearsal=float(rehearsal.detach()),gradient_norm=float(norm),communication_seconds=communication,
                learning_rates={g['category']:g['lr'] for g in opt.param_groups},peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            if balance:row['gradient_balance']={k:float(torch.stack([x[k] for x in balance]).mean()) for k in balance[0]}
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if rank==0 and (step<3 or (step+1)%25==0):print(json.dumps(row),flush=True)
            if (step+1)%c['serial_completion']['checkpoint_every']==0 or step+1==stop:save(out/'last.pt',model,opt,scheduler,step+1,c,provenance)
    if rank==0:atomic_json(out/'completion.json',dict(completed=True,step=stop,preflight=a.preflight,world=world,gradient_path_norms=path_norms,checkpoint_sha256=sha(out/'last.pt'),official_test_access=False))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
