"""Bounded clean two-stream training; no experiment-weight migration."""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/dexycb_fp_v3_inductor')
os.environ.setdefault('TRITON_CACHE_DIR','/tmp/dexycb_fp_v3_triton')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np,torch,torch.distributed as dist,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.optimized_training import OptimizedEpisode,make_optimizer,synchronize_gradients
from lip.unified.checkpoint import save,resume,atomic_json
from lip.engine.jepa_checkpoint import sha,restore_rng
from lip.engine.stream_checkpoint import source_hash
from lip.engine.config import check_data_gate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stop-at',type=int,required=True);parser.add_argument('--resume',type=Path)
    parser.add_argument('--repair-from',type=Path)
    parser.add_argument('--curriculum-from',type=Path)
    parser.add_argument('--extend-from',type=Path)
    parser.add_argument('--history-from',type=Path)
    parser.add_argument('--performance-from',type=Path)
    a=parser.parse_args();c=yaml.safe_load(a.config.read_text());rt=c['runtime'];t=c['training']
    if sum(v is not None for v in (a.resume,a.repair_from,a.curriculum_from,a.extend_from,a.history_from,a.performance_from))>1:raise ValueError('Choose one resume/adaptation path')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if c['architecture_id'] not in ('stream_two_input_jepa_v9','stream_conv_cross_jepa_v10','stream_conv_cross_geohistory_jepa_v10','stream_conv_cross_supported_history_jepa_v10','stream_conv_cross_dense_history_jepa_v10','stream_recovered_relation_jepa_v11','stream_cad_surface_jepa_v12'):raise ValueError('Two-stream architecture required')
    if a.repair_from or a.curriculum_from:raise ValueError('Clean training forbids migration flags')
    if world!=rt['world'] or world*rt['microbatch']!=t['effective_batch'] or rt['accumulation']!=1:raise ValueError('Fixed effective batch contract')
    start=c['migration']['source_step']
    if 'horizon_continuation' in c and not (a.extend_from or a.resume or a.performance_from):raise ValueError('Extension requires an explicit resume source')
    if a.stop_at>t['max_steps'] or a.stop_at<=start:raise ValueError('Explicit stop must fit remaining declared budget')
    torch.cuda.set_device(local);torch.set_num_threads(2)
    if torch.cuda.device_count()!=1:raise ValueError('Launch v3 workers through tools/fp_worker.py (one visible GPU per worker)')
    torch.use_deterministic_algorithms(rt.get('deterministic_algorithms',True))
    torch.utils.deterministic.fill_uninitialized_memory=rt.get('fill_uninitialized_memory',True)
    random.seed(c['seed']+rank);np.random.seed(c['seed']+rank);torch.manual_seed(c['seed'])
    if world>1:dist.init_process_group('nccl')
    model=build_model(c)
    v11_source=None;v11_receipt=None;reference_model=None;recovery_source=None;recovery_receipt=None
    if 'reconstruction_only' in c:
        if 'v11_initialization' in c or any(v is not None for v in (a.repair_from,a.curriculum_from,a.history_from)) or (a.extend_from and 'horizon_continuation' not in c):
            raise ValueError('Recovery-only continuation cannot combine migrations')
        from lip.unified.reconstruction_only import initialize_training
        recovery_source,recovery_receipt,reference_model=initialize_training(model,c,world)
    elif 'v11_initialization' in c:
        if any(v is not None for v in (a.repair_from,a.curriculum_from,a.extend_from,a.history_from)):
            raise ValueError('V11 initialization cannot combine with other migrations')
        from lip.unified.v11_initialization import initialize_training
        v11_source,v11_receipt=initialize_training(model,c,world)
    factory=Factory(c,model,make_store(c,model))
    optimizer=make_optimizer(model,c,fused=rt['fused_optimizer'])
    def rate(offset):
        if 'horizon_continuation' in c:
            from lip.unified.horizon_resume import extension_factor,scheduler_origin
            h=c['horizon_continuation']
            return extension_factor(offset+scheduler_origin(c),h['source_step'],h['rewarm_steps'],t['max_steps'],h['floor'])
        if 'reconstruction_only' in c:
            from lip.unified.horizon_resume import extension_factor
            h=c['reconstruction_only'];start=c['migration']['source_step']
            return extension_factor(offset+start,start,h['warmup_steps'],t['max_steps'],.1)
        if 'v11_initialization' in c:
            from lip.unified.horizon_resume import extension_factor
            h=c['v11_initialization'];start=c['migration']['source_step']
            return extension_factor(offset+start,start,h['warmup_steps'],t['max_steps'],.1)
        if rt.get('phase_relative_schedule',False):
            progress=min(1,offset/max(1,t['max_steps']-c['migration']['source_step']))
            return .1+.9*.5*(1+math.cos(math.pi*progress))
        step=c['migration']['source_step']+offset
        if step<t['lr_warmup_steps']:return (step+1)/t['lr_warmup_steps']
        progress=min(1,(step-t['lr_warmup_steps'])/max(1,t['max_steps']-t['lr_warmup_steps']))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    audit=check_data_gate(c['paths']['index_root'])
    provenance=dict(source_sha256=source_hash(),entrypoint_sha256=sha(__file__),weights=c['weights'],
        split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],initializers_sha256=factory.initializers_sha256,
        migration=model.migration,training_split='train',official_test_access=False,
        cuda_rng_mapping='saved per-rank current-device generator maps to worker-local CUDA0; v3 resume keeps visibility fixed',
        online_geometry=model.migration.get('pair_encoding','random trainable nine-channel geometry CNN'),static_geometry='original sealed complete-CAD Utonia cache',
        reduction='one FP32 packed all-reduce after whole-episode gradient accumulation',
        teacher='natural-hidden full-CAD; artificial-hidden real RGB-D; unmasked visible no completion loss')
    if 'performance_resume' in c:provenance['performance_resume']=c['performance_resume']
    if 'pose_repair' in c:provenance['pose_repair']=c['pose_repair']
    if 'curriculum_continuation' in c:provenance['curriculum_continuation']=c['curriculum_continuation']
    if 'horizon_continuation' in c:provenance['horizon_continuation']=c['horizon_continuation']
    if 'history_repair' in c:provenance['history_repair']=c['history_repair']
    if v11_source is not None:
        for key in ('split_hash','mesh_hash','initializers_sha256','weights'):
            if v11_source['provenance'][key]!=provenance[key]:raise ValueError('V11 source data mismatch: '+key)
        provenance['v11_initialization']=c['v11_initialization']
    if recovery_source is not None:
        for key in ('split_hash','mesh_hash','initializers_sha256','weights'):
            if recovery_source['provenance'][key]!=provenance[key]:raise ValueError('Recovery-only source data mismatch: '+key)
        provenance['reconstruction_only']=c['reconstruction_only']
    out=Path(c['paths']['output'])
    adaptation=None
    if a.resume:
        record=resume(a.resume,model,optimizer,scheduler,c,provenance,rank,world);start=record['step']
    elif a.performance_from:
        from lip.unified.encoder_performance import resume_performance
        record=resume_performance(a.performance_from,model,optimizer,scheduler,c,rank,world)
        start=record["step"];adaptation=dict(kind="performance_only",optimizer_reset=False,rng_reset=False)
    elif a.extend_from:
        from lip.unified.horizon_resume import extend_horizon
        start,adaptation=extend_horizon(a.extend_from,model,optimizer,scheduler,c,provenance,rank,world)
    elif v11_source is not None:
        adaptation=v11_receipt  # New seed42 stage; do not inherit parent RNG/sampler.
    elif recovery_source is not None:
        adaptation=recovery_receipt
        restore_rng(recovery_source['rng'][rank])
    elif a.history_from:
        from lip.unified.history_repair import adapt_history
        start,adaptation=adapt_history(a.history_from,model,optimizer,scheduler,c,rank,world)
    elif a.repair_from:
        from lip.unified.pose_repair import adapt
        start,adaptation=adapt(a.repair_from,model,optimizer,scheduler,c,rank,world,rate)
    elif a.curriculum_from:
        from lip.unified.curriculum_resume import adapt_curriculum
        start,adaptation=adapt_curriculum(a.curriculum_from,model,optimizer,scheduler,c,rank,world)
    else:
        if 'horizon_continuation' in c:raise ValueError('Extension config requires --extend-from or --resume')
        if start!=0:raise ValueError('Fresh initialization must start at step zero')
    if a.stop_at<=start:raise ValueError('Non-increasing stop')
    del v11_source,recovery_source
    # New projection initialization and all migrated weights must agree before
    # using a manual reducer, which intentionally has no DDP initial broadcast.
    digest=hashlib.sha256()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:digest.update(name.encode());digest.update(parameter.detach().cpu().numpy().tobytes())
    hashes=[digest.hexdigest()]
    if world>1:
        hashes=[None]*world;dist.all_gather_object(hashes,digest.hexdigest())
    if len(set(hashes))!=1:raise ValueError('Ranks disagree on initial trainable weights')
    if rank==0:
        out.mkdir(parents=True,exist_ok=bool(a.resume))
        if not a.resume:atomic_json(out/'manifest.json',dict(completed=False,config=c,provenance=provenance,
            migration_source_step=start,optimizer_updates_since_migration=0,default_model_changed=False,pose_repair_optimizer=adaptation,initial_trainable_sha256=hashes[0]))
    if world>1:dist.barrier()
    if not a.resume:
        save(out/('extension_start.pt' if a.extend_from else 'initial.pt'),model,optimizer,scheduler,start,c,provenance)
    episode=OptimizedEpisode(model,factory.renderer,c,reference_model=reference_model)
    if rt.get('reference_dino_graph') or rt.get('teacher_dino_graph'):
        from lip.unified.execution_speed import prime_execution_graphs
        prime_execution_graphs(episode)
    ids=lambda step:[c['seed']*1000033+(step*world+rank)*rt['microbatch']+i for i in range(rt['microbatch'])]
    factory.prefetch_rigid(ids(start),supervised=t['episode_frames'],burn=0)
    with (out/f'rank{rank}.jsonl').open('a') as log:
        for step in range(start,a.stop_at):
            tick=time.perf_counter();pairs=[factory.sample(seed) for seed in ids(step)];inputs,targets=zip(*pairs)
            if step+1<a.stop_at:factory.prefetch_rigid(ids(step+1),supervised=t['episode_frames'],burn=0)
            model.weights_version=f'{model.architecture_id}/{c["seed"]}/{step}'
            optimizer.zero_grad(set_to_none=True)
            loss,parts=episode(inputs,targets,segmented=rt['segmented_backward'])
            with episode.profile.record('gradient_communication'):gradient_bytes=synchronize_gradients(model.parameters())
            with episode.profile.record('clip_and_optimizer'):
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None],1.,foreach=True)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient; checkpoint not advanced')
                optimizer.step();scheduler.step()
            if getattr(model, "trainable_encoder", False):
                from lip.unified.ema_encoder import update_ema
                with episode.profile.record("ema_update"):
                    update_ema(model)
            statistics=torch.cat((loss[None],parts))
            if rt.get('history_pair_frames'):statistics=torch.cat((statistics,episode.history_pair_metrics))
            if rt.get('pose_pair_frames'):statistics=torch.cat((statistics,episode.pose_pair_metrics))
            if c.get('recovery_focus',{}).get('enabled',False):statistics=torch.cat((statistics,episode.history_supported_target_fraction.detach()[None]))
            if world>1:dist.all_reduce(statistics);statistics/=world
            torch.cuda.synchronize()
            row=dict(step=step+1,phase='joint',loss=float(statistics[0]),metrics=statistics[1:].tolist(),
                learning_rates=[g['lr'] for g in optimizer.param_groups],
                seconds=time.perf_counter()-tick,component_cuda_seconds=episode.profile.seconds(),gradient_bytes=gradient_bytes,
                effective_episodes=t['effective_batch'],episode_frames=t['episode_frames'],sampler_position=(step+1)*t['effective_batch'],
                optimizer_updates_since_migration=step+1-c['migration']['source_step'],migration_source_step=c['migration']['source_step'])
            if 'horizon_continuation' in c:row['horizon_updates']=step+1-c['horizon_continuation']['source_step']
            if getattr(model, 'trainable_encoder', False):
                row.update(ema_updates=int(model.ema_updates),ema_momentum=model.ema_momentum,encoder_trainable=True)
            if 'reconstruction_only' in c:
                row.update(phase='jepa_recovery_only',pose_loss_enabled=False,pose_metrics_computed=False,
                    pose_parameters_frozen=True,crop_reference_sha256=c['reconstruction_only'].get('reference',{}).get('sha256',c['reconstruction_only']['source_sha256']))
            if c.get('recovery_focus',{}).get('enabled',False):
                from lip.unified.losses import RECONSTRUCTION_METRICS
                from lip.unified.recovery_focus import FOCUS_METRICS
                names=('pose','translation','rotation','points')+RECONSTRUCTION_METRICS+FOCUS_METRICS
                if c.get('cad_surface'):names+=('cad_surface_real','cad_surface_proxy')
                if t['loss_weights'].get('log_feature_layers',False):
                    names+=('real_feature_mid','real_feature_last','proxy_feature_mid','proxy_feature_last')
                    row['dino_layers']=c['dino_layers']
                    row['feature_layer_weights']=[t['loss_weights']['feature_mid_weight'],t['loss_weights']['feature_last_weight']]
                if t['loss_weights'].get('local_difference',0) or t['loss_weights'].get('local_correspondence',0):
                    from lip.unified.local_structure import LOCAL_METRICS
                    names += LOCAL_METRICS
                row['recovery_metrics']={k:float(statistics[i+1]) for i,k in enumerate(names)}
                row['recovery_focus']=c['recovery_focus']['version']
                row['history_supported_target_fraction']=float(statistics[-1])
            if rt.get('jepa_pose_geometry',False):row['pose_error_aux_loss']=float(statistics[len(parts)])
            if rt.get('pose_pair_frames'):
                row['pose_pair_response_loss']=float(statistics[-2]);row['pose_pair_absolute_loss']=float(statistics[-1])
            log.write(json.dumps(row)+'\n');log.flush()
            if rank==0:print(json.dumps(row),flush=True)
            if (step+1)%rt['checkpoint_every']==0 or step+1==a.stop_at:
                save(out/'last.pt',model,optimizer,scheduler,step+1,c,provenance)
    if rank==0:atomic_json(out/'chunk_receipt.json',dict(completed=True,start_step=start,step=a.stop_at,checkpoint_sha256=sha(out/'last.pt')))
    if world>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
