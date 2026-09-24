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
    del source
    assert model.surface_decoder_kind=='dpt' and model.cad_rope3d_enabled
    assert not hasattr(reference,'surface_decoder_kind') and not getattr(reference,'cad_rope3d_enabled',False)
    teacher_before=digest(model.ema_teacher);reference_before=digest(reference)
    factory=Factory(c,model,make_store(c,model))
    episode=OptimizedEpisode(model,factory.renderer,c,reference_model=reference)
    prime_execution_graphs(episode)
    inputs,targets=zip(*(factory.sample(42000139+i,frames=40) for i in range(4)))
    tick=time.monotonic();loss,metrics=episode(inputs,targets);torch.cuda.synchronize()
    first_seconds=time.monotonic()-tick
    model.zero_grad(set_to_none=True)
    tick=time.monotonic();loss,metrics=episode(inputs,targets);torch.cuda.synchronize()
    seconds=time.monotonic()-tick
    assert torch.isfinite(loss) and all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    names=['cad_surface.rope3d.gain','surface_head.output.6.weight']
    names += [f'surface_head.projects.{i}.weight' for i in range(4)]
    names += [f'core.blocks.{i}.spatial.q.weight' for i in range(4)]
    names += ['encoder.backbone.patch_embed.proj.weight','encoder.backbone.blocks.10.mlp.fc1.weight']
    parameters=dict(model.named_parameters())
    grads={n:float(parameters[n].grad.norm()) for n in names}
    assert all(v>0 for v in grads.values())
    assert all(p.grad is None for p in model.ema_teacher.parameters())
    assert all(p.grad is None for p in model.writer.parameters())
    assert all(p.grad is None for n,p in model.named_parameters() if is_pose_parameter(n))
    assert digest(model.ema_teacher)==teacher_before and digest(reference)==reference_before
    # Staged geometry diagnostics, when enabled, follow normal diagnostics.
    from lip.unified.surface_normals import NORMAL_METRICS
    tail=0;staged_metrics={}
    if c.get('staged_rope',{}).get('enabled'):
        from lip.unified.staged_rope import STAGED_METRICS
        tail=len(STAGED_METRICS)
        staged_metrics={n:float(v) for n,v in zip(STAGED_METRICS,metrics[-tail:])}
        assert staged_metrics['rope_measured_fraction']>0 and staged_metrics['rope_recovered_fraction']>0
        assert staged_metrics['rope_recovered_trust']<=c['staged_rope']['recovered_max_trust']
    normal_metrics={n:float(v) for n,v in zip(NORMAL_METRICS,metrics[len(metrics)-tail-len(NORMAL_METRICS):len(metrics)-tail])}
    assert normal_metrics['normal_valid_fraction_real']>0 and normal_metrics['normal_valid_fraction_proxy']>0
    assert all(g['category']=='new' for g in optimizer.param_groups if any(n.startswith('surface_head.') for n in g['names']))
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None],1.,foreach=True)
    optimizer.step();update_ema(model)
    assert int(model.ema_updates)==before_ema+1 and model.cad_surface.rope3d.gain.abs().sum()>0
    state={k:v.detach().cpu().clone() for k,v in core_state(model).items()}
    with torch.no_grad():model.surface_head.output[-1].weight.add_(1)
    load_core(model,state);assert all(torch.equal(v.cpu(),state[k]) for k,v in core_state(model).items())
    result=dict(passed=True,config_sha256=sha(a.config),source_sha256=c['reconstruction_only']['source_sha256'],
        batch=4,frames=40,loss=float(loss),seconds=seconds,first_seconds_including_compile=first_seconds,
        component_cuda_seconds=episode.profile.seconds(),gradient_norms=grads,normal_metrics=normal_metrics,
        staged_metrics=staged_metrics,optimizer_preserved_exactly=preserve,
        crop_reference_unchanged=True,teacher_no_gradient=True,pose_no_gradient=True,history_no_gradient=True,
        shared_source_exact=True,full_model_restore_exact=True,ema_source_updates=before_ema,
        persisted_optimizer_updates=0,discarded_preflight_updates=1,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
    out=Path(__file__).resolve().parents[1]/'preflight';out.mkdir(exist_ok=True)
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
