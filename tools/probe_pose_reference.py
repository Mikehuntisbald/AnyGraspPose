"""Real-train equivalence and bounded learning check of retained-pose feedback."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import load_stream_config, make_model
from lip.engine.stream_checkpoint import load_init, sha, source_hash
from lip.engine.stream_training import StreamTrainingModule
from lip.engine.config import check_data_gate
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ('parent-experiment', 'manifests', 'data-root', 'index-root', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    a = p.parse_args();torch.set_num_threads(2);torch.manual_seed(42);a.out.mkdir(parents=True, exist_ok=False)
    e = json.loads((a.parent_experiment/'experiment.json').read_text())
    parent = a.parent_experiment/'spatial/train/last.pt'
    expected = '3371614e5efa788bf8c64cd10b64498040edf68a49d246c136d779e096ee0f2c'
    assert sha(parent) == expected
    c = load_stream_config(e['arms']['spatial']['config']);base = make_model(c).cuda().eval()
    load_init(parent, base, check_data_gate(a.index_root), c)
    for parameter in base.parameters(): parameter.requires_grad_(False)
    parent_parameters = {name: t.detach().clone() for name, t in base.state_dict().items()}
    manifests = {name: json.loads((a.manifests/(name+'.json')).read_text()) for name in ('control', 'natural')}
    selections = list(map(json.loads, (a.manifests/'selections.jsonl').read_text().splitlines()))
    ids = [i for i, row in enumerate(selections) if row['hard_selected']][:2]
    datasets = {name: StreamClips(a.data_root, a.index_root, 8, 48, fixed=manifest) for name, manifest in manifests.items()}
    groups = {'uniform': [datasets['control'][i] for i in (0, 1)],
              'natural_hard': [datasets['natural'][i] for i in ids]}
    renderer = Renderer('cuda');baseline = {}
    with torch.no_grad():
        for name, samples in groups.items():
            for precision in ('fp32', 'bf16'):
                baseline[(name, precision)] = StreamTrainingModule(base, dict(c, precision=precision), renderer)(samples, True)['predictions'].cpu()
    del base
    c = dict(c, architecture_id='stream_rk_pose_reference');model = make_model(c).cuda().eval()
    loaded = model.load_state_dict(parent_parameters, strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('pose_reference_feedback.') for k in loaded.missing_keys)
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name.startswith('pose_reference_feedback.'))
    equivalence = {}
    with torch.no_grad():
        for name, samples in groups.items():
            equivalence[name] = {}
            for precision in ('fp32', 'bf16'):
                out = StreamTrainingModule(model, dict(c, precision=precision), renderer)(samples, True)
                prediction = out['predictions'].cpu()
                assert torch.equal(prediction, baseline[(name, precision)])
                assert torch.count_nonzero(out['reference_diagnostics'][:4]) == 0
                equivalence[name][precision] = dict(bitwise_equal=True, frames=prediction.shape[0]*prediction.shape[1])
    runner = StreamTrainingModule(model, c, renderer);model.train()
    optimizer = torch.optim.AdamW(model.pose_reference_feedback.parameters(), lr=5e-5, weight_decay=.05);rows = []
    for step, name in enumerate(('uniform', 'natural_hard', 'uniform', 'natural_hard'), 1):
        optimizer.zero_grad(set_to_none=True);out = runner(groups[name], True);out['loss'].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.pose_reference_feedback.parameters(), 1.)
        assert torch.isfinite(grad) and grad > 0
        output_grad = model.pose_reference_feedback.readout[-1].weight.grad
        assert output_grad is not None and torch.isfinite(output_grad).all() and (output_grad.abs().sum(1) > 0).all()
        assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith('pose_reference_feedback.'))
        assert torch.isfinite(out['predictions']).all() and torch.isfinite(out['loss'])
        rotation = out['predictions'][..., :3, :3]
        assert (rotation.transpose(-1, -2) @ rotation - torch.eye(3, device='cuda')).abs().max() < 1e-3
        assert (out['predictions'][..., 2, 3] > .01).all()
        optimizer.step()
        rows.append(dict(step=step, population=name, loss=float(out['loss'].detach()), grad_norm=float(grad),
            supervised_frames=out['supervised_frames'], reference_diagnostics=out['reference_diagnostics'].cpu().tolist()))
    assert all(torch.equal(t, model.state_dict()[name]) for name, t in parent_parameters.items())
    assert (model.pose_reference_feedback.readout[-1].weight.abs().sum(1) > 0).all()
    torch.save(dict(state_dict=model.pose_reference_feedback.state_dict(), parent_sha256=expected, prototype_only=True), a.out/'prototype_head.pt')
    report = dict(passed=True, scope='Registered pose-reference tracker. Four real train fragments and four head-only optimizer steps; no val/test, FP, or accuracy-gain claim.',
        source_sha256=source_hash(), parent_sha256=expected, config=c, equivalence=equivalence, train_steps=rows,
        all_parent_tensors_unchanged=True, rotation_and_center_output_gradients_nonzero=True,
        real_zero_residual_equivalence_frames=224, samples={name: [s['sample'] for s in samples] for name, samples in groups.items()},
        manifest_sha256={name: sha(a.manifests/(name+'.json')) for name in manifests},
        reference_source='Provided estimate at first update; never refreshed by actor predictions; first-update residence clock.',
        precision='Network BF16; reference action, pose update and loss FP32', head_sha256=sha(a.out/'prototype_head.pt'),
        formal_training_approved=False)
    (a.out/'probe.json').write_text(json.dumps(report, indent=2));print(json.dumps({k: v for k, v in report.items() if k != 'config'}, indent=2))


if __name__ == '__main__': main()
