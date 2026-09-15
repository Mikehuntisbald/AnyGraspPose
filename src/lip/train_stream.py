import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from lip.engine.config import check_data_gate
from lip.engine.stream_config import load_stream_config,make_model,optimizer_and_scheduler,verify_preflight
from lip.engine import stream_checkpoint as checkpoint
from lip.engine.stream_training import StreamTrainingModule
from lip.geometry.renderer import Renderer
from lip.data.stream_clips import StreamClips,collate
from lip.data.stream_sampling import load_fixed_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--init-from');group.add_argument('--resume')
    p.add_argument('--data-root',default=os.environ.get('DEX_YCB_DIR'));p.add_argument('--index-root',default='cache/dexycb_s0')
    p.add_argument('--max-steps',type=int);p.add_argument('--preflight',action='store_true')
    p.add_argument('--fixed-manifest');a=p.parse_args()
    if a.max_steps is not None and a.max_steps<1:p.error('--max-steps must be positive; zero must never expand into a long run')
    if not a.data_root:p.error('Set DEX_YCB_DIR or --data-root')
    c=load_stream_config(a.config);world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    fixed,sampling_contract=load_fixed_manifest(a.fixed_manifest,c.get('fixed_sampling_manifest_sha256'))
    if world!=c['world_size']:raise ValueError('Config world_size must equal the actual launch world size')
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if c['precision']=='bf16' and device.type!='cuda':raise ValueError('BF16 preflight requires a GPU')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed']+rank);random.seed(c['seed']+rank);np.random.seed(c['seed']+rank)
    audit=check_data_gate(a.index_root)
    if a.preflight:
        if a.max_steps is None or a.max_steps>350:raise ValueError('Preflight bypass is limited to explicit <=350 stage steps')
    else:verify_preflight(c,audit)
    output=Path(a.output)
    if rank==0:
        if not a.resume and output.exists() and any(output.iterdir()):raise FileExistsError('New stage refuses to overwrite '+str(output))
        output.mkdir(parents=True,exist_ok=True)
        (output/'config.json').write_text(json.dumps(c,indent=2))
        (output/'sampling_contract.json').write_text(json.dumps(sampling_contract,indent=2))
    if world>1:dist.barrier()
    model=make_model(c).to(device)
    if a.resume:
        header=torch.load(a.resume,map_location='cpu',weights_only=False);model.migration_status=header.get('migration_status',{})
        parent=header.get('parent');del header
    else:
        header=checkpoint.load_init(a.init_from,model,audit,c)
        parent=dict(path=str(Path(a.init_from).resolve()),sha256=checkpoint.sha(a.init_from),
                    new_stage_step=header.get('new_stage_step',0),ancestor=header.get('parent'));del header
    if c.get('freeze_rgb_for_stage',False):model.rgb.requires_grad_(False)
    optimizer,scheduler=optimizer_and_scheduler(model,c)
    wrapper=StreamTrainingModule(model,c,Renderer(device))
    runner=DDP(wrapper,device_ids=[local] if device.type=='cuda' else None,broadcast_buffers=False,find_unused_parameters=False) if world>1 else wrapper
    # Restore AFTER DDP construction, model initialization and optimizer creation consume RNG.
    step=0;position=0
    if a.resume:
        saved=checkpoint.resume(a.resume,model,optimizer,scheduler,audit,c,rank,world)
        step=saved['new_stage_step'];position=saved['sampler_position']
        # Verify restored RNG before creating the dedicated DataLoader generator.
        assert torch.equal(torch.get_rng_state(),saved['rng'][rank]['torch'].cpu())
        assert scheduler.last_epoch==step and len(optimizer.state_dict()['state'])==len(saved['optimizer']['state'])
        (output/f'resume_rank{rank}.json').write_text(json.dumps(dict(passed=True,checkpoint_sha256=checkpoint.sha(a.resume),
            loaded_step=step,scheduler_step=scheduler.last_epoch,sampler_position=position,
            sampling_contract=sampling_contract,
            rng_restored=True,optimizer_state_entries=len(optimizer.state_dict()['state']),parameter_groups_verified=True),indent=2))
        del saved
    stop=min(a.max_steps or c['max_stage_steps'],c['max_stage_steps'])
    batch=c['batch_sequences_per_gpu'];accum=c['grad_accum_steps'];effective=batch*accum*world
    dataset=StreamClips(a.data_root,a.index_root,c['burn_in_frames'],c['supervised_unroll_frames'],seed=c['seed'],
        length=max(1,(stop-step)*batch*accum),start_sample=position,rank=rank,world=world,fixed=fixed,decode_threads=c['decode_threads'],
        external_initializers=c.get('external_initializers'),external_initializers_sha256=c.get('external_initializers_sha256'),
        real_initialization_probability=c.get('real_initialization_probability',0.),include_initial_observation=c.get('prime_initial_observation',False))
    kwargs=dict(batch_size=batch,collate_fn=collate,num_workers=c['num_workers'],pin_memory=c.get('pin_memory',False))
    if c['num_workers']:kwargs.update(prefetch_factor=c['prefetch_factor'],persistent_workers=True)
    # Dedicated generator prevents DataLoader setup from consuming training/dropout RNG on resume.
    kwargs['generator']=torch.Generator().manual_seed(c['seed'])
    loader=iter(DataLoader(dataset,**kwargs));model.train()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    with (output/f'rank{rank}.jsonl').open('a') as log:
        while step<stop:
            started=time.perf_counter();optimizer.zero_grad(set_to_none=True);logs=[];loss_log=[];count=0;alignment_logs=[]
            initialization_counts={key:0 for key in ('real_initializations','real_initializations_requested','real_initializations_missing','primed_observations')}
            startup_count=0;omitted_count=0
            for micro in range(accum):
                samples=next(loader)
                context=runner.no_sync() if world>1 and micro<accum-1 else nullcontext()
                with context:
                    result=runner(samples);loss=result['loss']/accum;loss.backward()
                logs.append(result['metrics']);loss_log.append(result['loss'].detach());count+=result['supervised_frames']
                if 'spatial_alignment_diagnostics' in result:alignment_logs.append(result['spatial_alignment_diagnostics'])
                startup_count+=result['startup_supervised_frames'];omitted_count+=result['omitted_late_targets']
                for key in initialization_counts:initialization_counts[key]+=result.get(key,0)
            # Fixed encoder weights and no accumulated Adam moments during freeze;
            # gradients were computed so DDP's graph/parameter set stays constant.
            if step<c['freeze_rgb_steps']:
                for parameter in model.rgb.parameters():parameter.grad=None
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip_norm'])
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite unroll gradient; no optimizer step or checkpoint committed')
            optimizer.step();scheduler.step();step+=1;position+=effective
            if device.type=='cuda':torch.cuda.synchronize(device)
            row=dict(new_stage_step=step,scheduler_step=scheduler.last_epoch,sampler_position=position,
                loss=float(torch.stack(loss_log).mean()),metrics=torch.stack(logs).mean(0).cpu().tolist(),
                actual_supervised_frames_rank=count,actual_supervised_frames_global=count*world,
                effective_sequences=effective,cache_bytes=result['cache_bytes'],kv_has_training_graph=result['kv_has_training_graph'],
                grad_norm=float(grad),seconds=time.perf_counter()-started,lrs=[g['lr'] for g in optimizer.param_groups],
                allocated=torch.cuda.memory_allocated(device) if device.type=='cuda' else None,
                peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            if 'cross_diagnostics' in result:
                row.update(zip(('cross_gate_mean','cross_update_norm','object_latent_norm'),result['cross_diagnostics'].cpu().tolist()))
                row['context_kv_has_training_graph']=result['context_kv_has_training_graph']
            if alignment_logs:
                row.update(zip(('alignment_ce','alignment_top1','alignment_visible_keys','alignment_supported_queries','alignment_eligible_clips'),torch.stack(alignment_logs).mean(0).cpu().tolist()))
            if 'rk_diagnostics' in result:
                row.update(zip(('observation_support_mean','anchors_read_mean'),result['rk_diagnostics'].cpu().tolist()))
                row.update(reliability_strength=float(model.reliability_strength.detach()),update_strength=float(model.update_strength.detach()),variant=model.variant)
            if 'spatial_diagnostics' in result:
                row.update(zip(('spatial_update_norm','spatial_tokens_read'),result['spatial_diagnostics'].cpu().tolist()))
            if 'aligned_diagnostics' in result:
                row.update(zip(('aligned_update_norm','aligned_tokens_read','aligned_queries'),result['aligned_diagnostics'].cpu().tolist()))
            if 'occlusion_diagnostics' in result:
                row.update(zip(('occlusion_pixel_fraction','occlusion_active_frame_fraction'),result['occlusion_diagnostics'].cpu().tolist()))
            if 'direct_pose_diagnostics' in result:
                row.update(zip(('direct_rotation_norm','direct_center_norm'),result['direct_pose_diagnostics'].cpu().tolist()))
            if 'reference_diagnostics' in result:
                row.update(zip(('reference_rotation_coefficient','reference_center_coefficient','reference_rotation_residual_norm','reference_center_residual_norm','reference_age_frames'),result['reference_diagnostics'].cpu().tolist()))
            if 'reference_write_diagnostics' in result:
                row.update(zip(('reference_write_rotation_coefficient','reference_write_center_coefficient','reference_write_rotation_norm','reference_write_center_norm','reference_write_valid_target'),result['reference_write_diagnostics'].cpu().tolist()))
                row['reference_has_training_graph']=result['reference_has_training_graph']
            if 'rotation_anchor_diagnostics' in result:
                row.update(zip(('rotation_anchor_fraction','rotation_anchor_gap_norm'),result['rotation_anchor_diagnostics'].cpu().tolist()))
            if 'smooth_rotation_diagnostics' in result:
                row.update(zip(('rotation_anchor_coefficient','rotation_anchor_abs_coefficient','rotation_anchor_positive_fraction',
                    'rotation_anchor_negative_fraction','rotation_anchor_gap_norm'),result['smooth_rotation_diagnostics'].cpu().tolist()))
            row['startup_supervised_frames_rank']=startup_count
            row['omitted_late_targets_rank']=omitted_count
            if c.get('prime_initial_observation',False):
                for key in ('real_initializations','real_initializations_requested','real_initializations_missing','primed_observations'):
                    row[key+'_rank']=initialization_counts[key]
            log.write(json.dumps(row)+'\n');log.flush()
            if step%c['save_every']==0 or step==stop:
                checkpoint.save(output/'last.pt',model,optimizer,scheduler,step,c,audit,position,parent)
        if rank==0:(output/'completed.json').write_text(json.dumps(dict(completed=True,new_stage_step=step,sampler_position=position,architecture_id=c['architecture_id']),indent=2))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
