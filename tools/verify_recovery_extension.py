"""Exact optimizer-preserving recovery horizon audit."""
import argparse, json, math, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.horizon_resume import exact, extension_factor, scheduler_origin
from lip.unified.reconstruction_only import is_pose_parameter
from lip.engine.jepa_checkpoint import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    c=yaml.safe_load(Path(a.config).read_text());h=c['horizon_continuation'];root=Path(c['paths']['output'])
    source=torch.load(h['source_checkpoint'],map_location='cpu',weights_only=False)
    initial=torch.load(root/'extension_start.pt',map_location='cpu',weights_only=False)
    last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler'):assert exact(initial[key],source[key]),key
    assert initial['step']==12000 and initial['optimizer']['state']
    assert last['step']>12000 and len(last['rng'])==8 and last['sampler_position']==last['step']*32
    updates=last['step']-12000
    assert int(last['model']['ema_updates'])==2250+updates
    for name,value in source['model'].items():
        if is_pose_parameter(name) or name.startswith('writer.') or '.history.' in name:
            assert torch.equal(value,last['model'][name]),name
    for prefix in ('encoder.','ema_teacher.','core.feature_mid.','core.feature_last.','cad_surface.'):
        assert any(not torch.equal(v,last['model'][k]) for k,v in source['model'].items() if k.startswith(prefix)),prefix
    assert last['scheduler']['last_epoch']==last['step']-scheduler_origin(c)
    for group,base in zip(last['optimizer']['param_groups'],last['scheduler']['base_lrs']):
        expected=base*extension_factor(last['step'],12000,100,25000,.1)
        assert math.isclose(group['lr'],expected,rel_tol=1e-10)
        assert not any(is_pose_parameter(n) or n.startswith('ema_teacher.') for n in group['names'])
    rank_steps=[]
    for rank in range(8):
        row=json.loads((root/f'rank{rank}.jsonl').read_text().splitlines()[-1]);rank_steps.append(row['step'])
        assert row['horizon_updates']==updates and row['pose_parameters_frozen'] and not row['pose_loss_enabled']
        assert row['dino_layers']==dict(student=[4,11],teacher=[4,11])
        assert all(math.isfinite(v) for v in row['recovery_metrics'].values())
    assert rank_steps==[last['step']]*8
    result=dict(passed=True,step=last['step'],additional_updates=updates,all_rank_steps=rank_steps,
        initial_model_optimizer_scheduler_exact=True,optimizer_reset=False,ema_updates=int(last['model']['ema_updates']),
        pose_history_unchanged=True,lr_schedule_verified=True,checkpoint_sha256=sha(root/'last.pt'))
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
