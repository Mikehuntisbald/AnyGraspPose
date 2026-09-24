"""Real 40-frame recovery backward, teacher isolation and exact EMA checks."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch, yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.reconstruction_only import initialize_training,is_pose_parameter
from lip.unified.training import Factory
from lip.unified.optimized_training import OptimizedEpisode,make_optimizer
from lip.unified.ema_encoder import update_ema
from lip.engine.jepa_checkpoint import sha,core_state,load_core


def digest(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():
        h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    c=yaml.safe_load(a.config.read_text());torch.set_num_threads(2);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    model=build_model(c);source,migration,reference=initialize_training(model,c,8);del source
    assert model.encoder.trainable_encoder and not model.ema_teacher.trainable_encoder
    assert model.encoder.feature_layers==(4,11) and model.ema_teacher.feature_layers==(4,11)
    assert reference.encoder.feature_layers==(6,12)
    assert int(model.ema_updates)==1250
    probe=torch.randn(2,3,224,224,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for encoder,layers in ((model.encoder,[3,10]),(model.ema_teacher,[3,10]),(reference.encoder,[5,11])):
            actual=encoder(probe)
            expected=encoder.backbone.get_intermediate_layers(probe,n=layers,reshape=False,norm=True)
            assert all(torch.equal(x,y) for x,y in zip(actual,expected))
    teacher_before=digest(model.ema_teacher);reference_before=digest(reference)
    factory=Factory(c,model,make_store(c,model))
    inputs,targets=zip(*(factory.sample(42000139+i,frames=40) for i in range(4)))
    run=OptimizedEpisode(model,factory.renderer,c,reference_model=reference)
    tick=time.monotonic();loss,metrics=run(inputs,targets);torch.cuda.synchronize()
    seconds=time.monotonic()-tick
    assert torch.isfinite(loss)
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    grads={n:float(p.grad.norm()) for n,p in model.encoder.named_parameters() if p.grad is not None}
    assert any(v>0 for k,v in grads.items() if 'patch_embed' in k)
    assert any(v>0 for k,v in grads.items() if 'blocks.10.' in k)
    assert all(p.grad is None for p in model.ema_teacher.parameters())
    assert all(p.grad is None for n,p in model.encoder.named_parameters() if 'blocks.11.' in n)
    assert all(p.grad is None for p in model.writer.parameters())
    assert model.core.feature_mid.weight.grad.norm()>0 and model.core.feature_last.weight.grad.norm()>0
    assert all(p.grad is None for n,p in model.named_parameters() if is_pose_parameter(n))
    assert digest(model.ema_teacher)==teacher_before and digest(reference)==reference_before
    optimizer=make_optimizer(model,c)
    assert any(g['category']=='encoder' for g in optimizer.param_groups)
    # Discarded in-memory update only, never persisted as training progress.
    name='backbone.blocks.10.mlp.fc1.weight'
    old=dict(model.ema_teacher.named_parameters())[name].detach().clone()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None],1.,foreach=True)
    optimizer.step()
    online=dict(model.encoder.named_parameters())[name].detach().clone()
    assert not torch.equal(old,online)
    update_ema(model)
    actual=dict(model.ema_teacher.named_parameters())[name]
    expected=old.mul(model.ema_momentum).add(online,alpha=1-model.ema_momentum)
    assert torch.equal(actual,expected)
    # Full checkpoint payload must preserve both branches, including update count.
    state={k:v.detach().cpu().clone() for k,v in core_state(model).items()}
    with torch.no_grad():next(model.encoder.parameters()).add_(1)
    load_core(model,state)
    assert all(torch.equal(v.cpu(),state[k]) for k,v in core_state(model).items())
    assert c['local_structure']['scales']==[1,2,4]
    result=dict(local_structure=c['local_structure'],passed=True,config_sha256=sha(a.config),source_sha256=c['reconstruction_only']['source_sha256'],
        student_layers=[4,11],teacher_layers=[4,11],reference_layers=[6,12],
        layer_selection_matches_official=True,layer_weights=[.625,.625],ema_source_updates=1250,
        both_recovery_heads_have_gradients=True,unused_block12_has_no_gradient=True,history_has_no_gradient=True,
        batch=4,frames=40,loss=float(loss),seconds=seconds,encoder_gradient_norms=grads,
        teacher_no_gradient=True,teacher_unchanged_before_ema=True,ema_formula_exact=True,
        crop_reference_unchanged=True,pose_no_gradient=True,full_state_restore_exact=True,
        persisted_optimizer_updates=0,discarded_preflight_updates=1,
        peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
    out=Path(__file__).resolve().parents[1]/'preflight';out.mkdir(exist_ok=True)
    (out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
