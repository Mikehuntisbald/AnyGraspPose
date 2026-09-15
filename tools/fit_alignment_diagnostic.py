"""Bounded train-only learnability check; output weights are never candidates."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lip.engine.stream_config import load_stream_config, make_model, optimizer_and_scheduler
from lip.engine.stream_checkpoint import load_init, sha, source_hash
from lip.engine.config import check_data_gate
from lip.engine.stream_training import StreamTrainingModule, supervision_positions
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import angle


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--arm', choices=('frozen_parent', 'joint'), required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / 'runs/train_diagnostic'; spec = json.loads((folder/'spec.json').read_text())
    out = folder / a.arm; out.mkdir(exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(20260915); random.seed(20260915); np.random.seed(20260915)
    assert sha(spec['initial_checkpoint']) == spec['initial_checkpoint_sha256']
    assert source_hash() == spec['source_sha256']
    c = load_stream_config(spec['configuration']); plan = spec['short_fit']
    c.update(max_stage_steps=plan['steps'], warmup_steps=plan['warmup'],
             lr_rotation_alignment=plan['branch_lr'], lr_loaded_modules=plan['loaded_lr'], lr_new_modules=plan['other_new_lr'])
    (out/'config.json').write_text(json.dumps(c, indent=2))
    model = make_model(c).cuda()
    saved = load_init(spec['initial_checkpoint'], model, check_data_gate(spec['index_root']), c)
    del saved
    initial = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    if a.arm == 'frozen_parent':
        model.requires_grad_(False); model.rotation_alignment.requires_grad_(True)
    else:
        model.rgb.requires_grad_(False)
    optimizer, scheduler = optimizer_and_scheduler(model, c)
    runner = StreamTrainingModule(model, c, Renderer('cuda'))
    dataset = StreamClips(spec['data_root'], spec['index_root'], 8, 48,
        fixed=json.loads(Path(spec['training_manifest']).read_text()), decode_threads=2,
        external_initializers=spec['train_initializers'], external_initializers_sha256=spec['train_initializers_sha256'],
        real_initialization_probability=.5, include_initial_observation=True)
    samples = {name: [dataset[v['manifest_index']] for v in rows] for name, rows in spec['cohort'].items()}
    assert all(s['real_initialization_requested'] and 'real_initial_pose' in s for ss in samples.values() for s in ss)
    assert not ({v['physical_sequence'] for v in spec['cohort']['fit']} & {v['physical_sequence'] for v in spec['cohort']['probe']})
    selected = torch.tensor(supervision_positions(8, 48, 8), device='cuda')
    evaluations = []

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        for name, ss in samples.items():
            corrections = []
            hook = model.rotation_alignment.register_forward_hook(lambda module, args, value: corrections.append(value.detach().float()))
            result = runner(ss, True); hook.remove()
            pred = result['predictions']; target = torch.stack([s['targets'] for s in ss]).cuda().float()
            rot = angle(pred[..., :3, :3] @ target[..., :3, :3].transpose(-1, -2)) * (180/torch.pi)
            center = (pred[..., :3, 3] - target[..., :3, 3]).norm(dim=-1) * 1000
            extra = torch.stack(corrections[1:], 1).norm(dim=-1) * (180/torch.pi)
            assert extra.shape == rot.shape == (8, 56)
            metrics = {}
            for pop, mask in [('supervised', selected), ('first8', torch.arange(56, device='cuda') < 8)]:
                metrics[pop] = {key: float(value[:, mask].mean()) for key, value in [('rotation_deg', rot), ('center_mm', center), ('branch_deg', extra)]}
                metrics[pop]['per_clip'] = [{key: float(value[i, mask].mean()) for key, value in [('rotation_deg', rot), ('center_mm', center), ('branch_deg', extra)]} for i in range(8)]
            row = dict(step=step, population=name, loss=float(result['loss']), metrics=metrics)
            evaluations.append(row); print(json.dumps(row), flush=True)
        model.train()

    evaluate(0)
    with (out/'steps.jsonl').open('x') as log:
        for step in range(1, plan['steps']+1):
            started = time.monotonic(); optimizer.zero_grad(set_to_none=True)
            result = runner(samples['fit']); loss = result['loss']; loss.backward()
            branch_grad = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in model.rotation_alignment.named_parameters()}
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip_norm'])
            assert torch.isfinite(grad) and torch.isfinite(loss)
            optimizer.step(); scheduler.step()
            row = dict(step=step, loss=float(loss.detach()), grad_norm=float(grad), branch_gradient_l2=branch_grad,
                       lrs=[g['lr'] for g in optimizer.param_groups], seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n'); log.flush()
            del result, loss
            if step in plan['evaluate_steps']: evaluate(step)
    changed = [n for n, v in model.state_dict().items() if not torch.equal(v.cpu(), initial[n])]
    assert changed and all(torch.isfinite(v).all() for v in model.state_dict().values())
    assert all(not n.startswith('rgb.') for n in changed)
    if a.arm == 'frozen_parent': assert all(n.startswith('rotation_alignment.') for n in changed)
    assert source_hash() == spec['source_sha256'] and sha(spec['initial_checkpoint']) == spec['initial_checkpoint_sha256']
    torch.save(dict(model=model.state_dict(), diagnostic_only=True, candidate=False, arm=a.arm, steps=plan['steps'], spec_sha256=sha(folder/'spec.json')), out/'diagnostic_only.pt')
    receipt = dict(completed=True, arm=a.arm, steps=plan['steps'], spec_sha256=sha(folder/'spec.json'),
        initial_checkpoint_sha256=spec['initial_checkpoint_sha256'], source_sha256=source_hash(), entrypoint_sha256=sha(Path(__file__)),
        final_diagnostic_sha256=sha(out/'diagnostic_only.pt'), evaluations=evaluations, changed_tensors=changed,
        rgb_bitwise_unchanged=True, parent_bitwise_unchanged=a.arm == 'frozen_parent', finite=True, val_test_access=False,
        optimizer_steps=plan['steps'], scope=plan['purpose'])
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__': main()
