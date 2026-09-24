"""Audit exact frozen tensors, optimizer ownership and all-rank recovery logs."""
import argparse, json, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lip.unified.reconstruction_only import is_pose_parameter
from lip.engine.jepa_checkpoint import sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path); p.add_argument('--out', type=Path); a = p.parse_args()
    root = Path(__file__).resolve().parents[1]; c = yaml.safe_load(a.config.read_text())
    directory = Path(c['paths']['output']); path = a.checkpoint or directory / 'last.pt'
    source = torch.load(c['reconstruction_only']['source_checkpoint'], map_location='cpu', weights_only=False)
    record = torch.load(path, map_location='cpu', weights_only=False)
    initial = torch.load(directory / 'initial.pt', map_location='cpu', weights_only=False)
    assert initial['step'] == source['step'] and not initial['optimizer']['state']
    assert all(torch.equal(v, initial['model'][k]) for k, v in source['model'].items())
    assert all(torch.equal(v, record['model'][k]) for k, v in source['model'].items() if is_pose_parameter(k))
    assert len(record['rng']) == 8 and record['sampler_position'] == record['step'] * 32
    names = [n for g in record['optimizer']['param_groups'] for n in g['names']]
    assert not any(is_pose_parameter(n) or n.startswith('ema_teacher.') for n in names)
    if c.get('ema_encoder',{}).get('enabled'):
        assert any(n.startswith('encoder.') for n in names)
        assert int(record['model']['ema_updates'])==int(initial['model']['ema_updates'])+record['step']-source['step']
        assert any(not torch.equal(v,record['model'][k]) for k,v in initial['model'].items() if k.startswith('encoder.'))
        assert any(not torch.equal(v,record['model'][k]) for k,v in initial['model'].items() if k.startswith('ema_teacher.'))
    else:
        assert not any(n.startswith('encoder.') for n in names)
    changed = [k for k in names if not torch.equal(record['model'][k], initial['model'][k])]
    for prefix in ('core.blocks.', 'core.feature_mid.', 'core.feature_last.', 'surface_head.')+(() if c['runtime'].get('disable_history',False) else ('writer.',)):
        assert any(k.startswith(prefix) for k in changed), prefix
    if c.get('cad_surface'):
        assert any(k.startswith('cad_surface.') for k in changed)
    if c['runtime'].get('disable_history',False):
        assert all(torch.equal(v,record['model'][k]) for k,v in source['model'].items() if k.startswith('writer.') or '.history.' in k)
    latest = []
    for rank in range(8):
        rows = [json.loads(v) for v in (directory / f'rank{rank}.jsonl').read_text().splitlines()]
        r = rows[-1]; latest.append(r['step'])
        assert r['phase'] == 'jepa_recovery_only' and not r['pose_loss_enabled'] and r['pose_parameters_frozen']
        if c.get('dino_layers'):
            assert r['dino_layers']==c['dino_layers'] and r['feature_layer_weights']==[.625,.625]
            d=r['recovery_metrics']
            assert abs(d['real_hidden_feature']-.625*(d['real_feature_mid']+d['real_feature_last']))<1e-5
            assert abs(d['cad_proxy_feature']-.625*(d['proxy_feature_mid']+d['proxy_feature_last']))<1e-5
            if c.get('local_structure'):
                from lip.unified.local_structure import LOCAL_METRICS
                import math
                assert all(math.isfinite(d[key]) for key in LOCAL_METRICS)
                assert d['local_difference_real_mid'] > 0 and d['local_difference_real_last'] > 0
    assert min(latest) >= record['step']
    result = dict(passed=True, completed=True, step=record['step'], checkpoint_sha256=sha(path),
        initial_core_exact=True, initial_optimizer_empty=True, frozen_pose_tensors_exact=True,
        optimizer_excludes_pose=True, changed_trainable_tensors=len(changed), all_rank_steps=latest,
        optimizer_updates_since_migration=record['step']-source['step'], sampler_position=record['sampler_position'])
    (a.out or root / 'startup_receipt.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result))


if __name__ == '__main__': main()
