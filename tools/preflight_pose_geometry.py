"""Real 40-frame H20 backward for DPT, RoPE and derived-normal supervision."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.reconstruction_only import initialize_training,is_pose_parameter
from lip.unified.training import Factory
from lip.unified.optimized_training import OptimizedEpisode,make_optimizer
from lip.unified.execution_speed import prime_execution_graphs
from lip.unified.ema_encoder import update_ema
from lip.engine.jepa_checkpoint import sha,core_state,load_core


def digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    c=yaml.safe_load(a.config.read_text());torch.set_num_threads(2);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    model=build_model(c);source,migration,reference=initialize_training(model,c,8)
    before_ema=int(model.ema_updates);assert before_ema==int(source['model']['ema_updates'])
    optimizer=make_optimizer(model,c)
    preserve=not c['reconstruction_only'].get('optimizer_reset',True)
    if preserve:
        from lip.unified.staged_rope import preserve_adam
        preserve_adam(optimizer,source)
        migration_adam=dict(entire_optimizer_exact=True)
    pose_before={n:p.detach().clone() for n,p in model.named_parameters() if is_pose_parameter(n)}
    del source
    assert model.surface_decoder_kind=='dpt' and model.cad_rope3d_enabled
    assert not hasattr(reference,'surface_decoder_kind') and not getattr(reference,'cad_rope3d_enabled',False)
    teacher_before=digest(model.ema_teacher);reference_before=digest(reference)
    factory=Factory(c,model,make_store(c,model))
    episode=OptimizedEpisode(model,factory.renderer,c,reference_model=reference)
    prime_execution_graphs(episode)
    inputs,targets=zip(*(factory.sample(42000139+i,frames=40) for i in range(4)))
    from lip.unified.features import prepare_scene,encode_scenes
    from lip.losses import pose_loss
    e=inputs[0]
    scene=prepare_scene(e.rgb[0],e.depth[0],e.initial,e.mesh,e.k,e.times[0],e.stream,e.cad, factory.renderer)
    obs=encode_scenes(model,[scene],cad_enabled=torch.ones(1,device='cuda',dtype=torch.bool),frame_id=0)
    from lip.unified.features import build_teachers
    from dataclasses import fields
    masks=[targets[0][1][0]];added=[torch.zeros_like(scene.bounds)]
    full=build_teachers(model.ema_teacher,[scene],targets[0][0][0:1],masks,added,factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True)
    lean=build_teachers(None,[scene],targets[0][0][0:1],masks,added,factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
    for f in fields(full):
        if f.name in ('real_mid','real_last','proxy_mid','proxy_last','proxy_rgb','real_rgb'):
            assert getattr(lean,f.name) is None
        elif isinstance(getattr(full,f.name),torch.Tensor):torch.testing.assert_close(getattr(lean,f.name),getattr(full,f.name),rtol=0,atol=0,equal_nan=True)
    def forbid_teacher(*args):raise AssertionError('DINO teacher was called during geometry-only training')
    teacher_hook=model.ema_teacher.register_forward_pre_hook(forbid_teacher)
    del full,lean
    compiled=model.compiled_frame;model.compiled_frame=None
    with torch.autocast('cuda',dtype=torch.bfloat16):output,_=model(obs,None,torch.zeros(1,device='cuda',dtype=torch.bool))
    points=torch.as_tensor(e.mesh['points'],device='cuda')[None]
    pose,_=pose_loss(output['pose_centered'],targets[0][0][0:1],points,obs.diameter)
    pose_paths=torch.autograd.grad(pose,(output['patch_latent'],output['surface_xyz'],output['surface_depth_residual']),allow_unused=False)
    pose_path_norms=[float(g.float().norm()) for g in pose_paths]
    assert all(v>0 and v<float('inf') for v in pose_path_norms)
    model.compiled_frame=compiled
    del output,obs,scene,pose_paths,pose
    model.zero_grad(set_to_none=True)
    tick=time.monotonic();loss,metrics=episode(inputs,targets);torch.cuda.synchronize()
    first_seconds=time.monotonic()-tick
    model.zero_grad(set_to_none=True)
    tick=time.monotonic();loss,metrics=episode(inputs,targets);torch.cuda.synchronize()
    seconds=time.monotonic()-tick
    assert torch.isfinite(loss) and all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    names=['cad_surface.rope3d.gain','surface_head.output.6.weight']
    names += [n for n,p in model.named_parameters() if is_pose_parameter(n) and p.requires_grad and (n=='query' or n.endswith('weight'))]
    names += [f'surface_head.projects.{i}.weight' for i in range(4)]
    names += [f'core.blocks.{i}.spatial.q.weight' for i in range(4)]
    names += ['encoder.backbone.patch_embed.proj.weight','encoder.backbone.blocks.10.mlp.fc1.weight']
    parameters=dict(model.named_parameters())
    grads={n:float(parameters[n].grad.norm()) for n in names}
    assert all(v>0 for v in grads.values())
    assert all(p.grad is None for p in model.ema_teacher.parameters())
    assert all(p.grad is None for p in model.writer.parameters())
    assert all(p.requires_grad for n,p in model.named_parameters() if is_pose_parameter(n))
    assert digest(model.ema_teacher)==teacher_before and digest(reference)==reference_before
    from lip.unified.pose_geometry import METRICS
    staged_metrics={n:float(v) for n,v in zip(METRICS,metrics)}
    assert staged_metrics['pose']>0 and staged_metrics['rope_recovered_fraction']>0
    for n,p in model.named_parameters():
        if n.startswith(('core.feature_mid.','core.feature_last.','core.log_error.')):assert p.grad is None
    normal_metrics={}
    assert all(g['category']=='new' for g in optimizer.param_groups if any(n.startswith('surface_head.') for n in g['names']))
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None],1.,foreach=True)
    optimizer.step();update_ema(model)
    assert any(not torch.equal(p,pose_before[n]) for n,p in model.named_parameters() if is_pose_parameter(n))
    assert int(model.ema_updates)==before_ema+1 and model.cad_surface.rope3d.gain.abs().sum()>0
    state={k:v.detach().cpu().clone() for k,v in core_state(model).items()}
    with torch.no_grad():model.surface_head.output[-1].weight.add_(1)
    load_core(model,state);assert all(torch.equal(v.cpu(),state[k]) for k,v in core_state(model).items())
    result=dict(passed=True,config_sha256=sha(a.config),source_sha256=c['reconstruction_only']['source_sha256'],
        batch=4,frames=40,teacher_dino_forwards_during_training=0,geometry_targets_identical=True,feature_head_gradients_absent=True,pose_to_patch_xyz_depth_gradient_norms=pose_path_norms,loss=float(loss),seconds=seconds,first_seconds_including_compile=first_seconds,
        component_cuda_seconds=episode.profile.seconds(),gradient_norms=grads,normal_metrics=normal_metrics,
        staged_metrics=staged_metrics,existing_optimizer_moments_preserved_exactly=preserve,optimizer_migration=migration_adam,
        crop_reference_unchanged=True,teacher_no_gradient=True,pose_has_gradient=True,history_no_gradient=True,
        shared_source_exact=True,full_model_restore_exact=True,ema_source_updates=before_ema,
        persisted_optimizer_updates=0,discarded_preflight_updates=1,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
    out=Path(c['paths']['output']).parents[1]/'preflight';out.mkdir(exist_ok=True)
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
