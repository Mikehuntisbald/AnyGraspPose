"""Audit exact frozen tensors, optimizer ownership and all-rank recovery logs."""
import argparse, json, sys, math
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
    assert initial['step'] == source['step']
    preserved_adam=not c['reconstruction_only'].get('optimizer_reset',True)
    if preserved_adam:
        from lip.unified.horizon_resume import exact
        if 'lr_intervention' in c:
            from copy import deepcopy
            check=deepcopy(initial['optimizer'])
            for group,saved in zip(check['param_groups'],source['optimizer']['param_groups']):group['lr']=saved['lr']
            assert exact(check,source['optimizer'])
        else:assert exact(initial['optimizer'],source['optimizer'])
    else:assert not initial['optimizer']['state']
    replaced = c.get('surface_decoder',{}).get('kind')=='dpt' and source['config'].get('surface_decoder',{}).get('kind')!='dpt'
    assert all(torch.equal(v, initial['model'][k]) for k, v in source['model'].items()
               if not (replaced and k.startswith('surface_head.')))
    if replaced:
        assert 'surface_head.3.weight' not in record['model']
        assert 'surface_head.output.6.weight' in record['model']
        assert 'cad_surface.rope3d.gain' in record['model']
        assert not initial['model']['cad_surface.rope3d.gain'].any()
        assert record['model']['cad_surface.rope3d.gain'].abs().sum()>0
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
    if c.get('staged_rope',{}).get('enabled'):
        assert set(initial['model'])==set(source['model'])
        assert all(torch.equal(v,initial['model'][k]) for k,v in source['model'].items())
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
            if c.get('staged_rope',{}).get('enabled'):
                from lip.unified.staged_rope import STAGED_METRICS
                assert all(math.isfinite(d[k]) for k in STAGED_METRICS)
                # Frames with all CAD references dropped have zero eligible
                # routing mass, so the episode average can sum to less than1.
                assert 0<=sum(d['rope_'+name+'_fraction'] for name in ('measured','recovered','fallback'))<=1+1e-5
    assert min(latest) >= record['step']
    result = dict(passed=True, completed=True, step=record['step'], checkpoint_sha256=sha(path),
        initial_shared_core_exact=True, replaced_surface_mlp=replaced, initial_optimizer_empty=not preserved_adam,
        initial_optimizer_preserved_exactly=preserved_adam and 'lr_intervention' not in c,
        initial_non_lr_optimizer_preserved_exactly=preserved_adam, frozen_pose_tensors_exact=True,
        optimizer_excludes_pose=True, changed_trainable_tensors=len(changed), all_rank_steps=latest,
        optimizer_updates_since_migration=record['step']-source['step'], sampler_position=record['sampler_position'])
    (a.out or root / 'startup_receipt.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result))


if __name__ == '__main__': main()
