"""V34: bounded geometry-only end-to-end training with exact continuation.

Full-stream single frames; same data, initialization, optimizer and objective
for the control, except the candidate's explicitly supervised CAD transport.
No pose/DINO losses, oracle rehearsal, GT student masks, or history.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from prepare_serial_completion import heldout
from lip.unified.build import build_model, make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene, encode_scenes, build_teachers
from lip.unified.execution_speed import crop_images_fast
from lip.unified.geometry_transport_objective import objective
from lip.unified.reconstruction_only import is_pose_parameter
from lip.unified.optimized_training import synchronize_gradients
from lip.unified.checkpoint import save, resume, atomic_json
from lip.unified.ema_encoder import update_ema
from lip.engine.jepa_checkpoint import sha
from lip.geometry.so3 import update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.manual_seed(config['seed']); random.seed(config['seed']+rank); np.random.seed(config['seed']+rank)
    if world > 1: dist.init_process_group('nccl')
    plan = config['geometry_transport_training']
    batch = config['runtime']['microbatch']
    assert world == 8 and world*batch == config['training']['effective_batch'] == 32
    model = build_model(config)
    if sha(plan['source_checkpoint']) != plan['source_sha256']:
        raise ValueError('Source geometry checkpoint changed')
    parent = torch.load(plan['source_checkpoint'], map_location='cpu', weights_only=False)
    if parent['step'] != plan['source_step']: raise ValueError('Source step mismatch')
    if config['cad_transport'].get('reference_conditioned'):
        key='cad_transport.head.0.weight';old=parent['model'][key]
        if old.shape[1]==16:
            extended=torch.zeros_like(model.state_dict()[key],device='cpu');extended[:,:16]=old
            parent['model'][key]=extended
        elif old.shape[1]!=25:raise ValueError('Unexpected transport channel count')
    status = model.load_state_dict(parent['model'], strict=False)
    if status.unexpected_keys or any(not n.startswith('cad_transport.') for n in status.missing_keys):
        raise ValueError('Unexpected geometry migration: '+str(status))
    if any(not torch.equal(model.state_dict()[n].cpu(), value.cpu()) for n, value in parent['model'].items()):
        raise ValueError('Existing model tensors changed during migration')
    del parent
    if plan.get('transport_gate_bias') is not None:
        with torch.no_grad():
            model.cad_transport.head[-1].bias[6] = plan['transport_gate_bias']
    model.fast_geometry = model.vector_geometry = model.trusted_training_inputs = True
    groups = {}
    frozen = []
    for name, parameter in model.named_parameters():
        active = not (is_pose_parameter(name) or name.startswith(('ema_teacher.', 'writer.', 'memory_position.', 'core.memory_', 'core.feature_', 'core.log_error.')) or '.history.' in name)
        if plan.get('decoder_only'):
            active = name.startswith('cad_transport.')
        if name.startswith('cad_transport.') and not config['cad_transport']['enabled']: active = False
        parameter.requires_grad_(active)
        if not active:
            frozen.append(name); continue
        kind = 'encoder' if name.startswith('encoder.') else ('new' if name.startswith(('surface_head.', 'cad_transport.')) else 'predictor')
        if name.startswith('cad_transport.') and 'transport' in config['training']['learning_rates']:kind='transport'
        group = groups.setdefault(kind, dict(params=[], names=[], category=kind, lr=config['training']['learning_rates'][kind]))
        group['params'].append(parameter); group['names'].append(name)
    optimizer = torch.optim.AdamW(list(groups.values()), weight_decay=.01, fused=True)
    maximum = plan['updates']; stop = args.stop_at or maximum
    if not 0 < stop <= maximum: raise ValueError('Budget exceeded')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step+1)/plan.get('warmup',50))*(.5+.25*(1+math.cos(math.pi*min(step, maximum)/maximum))))
    factory = Factory(config, model, make_store(config, model))
    root = Path(__file__).resolve().parents[1]
    source_files = {str(p.relative_to(root)): sha(p) for folder in ('src','tools','configs') for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    provenance = dict(source_sha256=hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest(),
                      source_checkpoint_sha256=plan['source_sha256'], source_step=plan['source_step'],
                      split_hash=factory.audit['split_hash'], mesh_hash=factory.audit['mesh_hash'],
                      initializers_sha256=factory.initializers_sha256, training_split='train',
                      official_test_access=False, teacher_input=False, pose_loss=False, dino_loss=False,
                      crop='Uniform full-stream frame; transported native error or training GT perturbation',
                      optimizer_reset=True, frozen_parameter_names=frozen)
    if plan.get('decoder_only'):
        provenance['decoder_only'] = True
        provenance['initial_gate_bias_override'] = plan.get('transport_gate_bias')
    if config['cad_transport'].get('reference_conditioned'):
        provenance['reference_conditioning']='9 explicit estimated-CAD geometry/residual channels; zero-expanded first convolution; all existing coefficients exact'
    out = Path(config['paths']['output'])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=bool(args.resume))
        atomic_json(out/'provenance.json', provenance)
    if world > 1: dist.barrier()
    torch.manual_seed(config['seed']+rank)
    start = 0
    if args.resume:
        record = resume(args.resume, model, optimizer, scheduler, config, provenance, rank, world)
        start = record['step']
        if rank == 0: atomic_json(out/f'resume{start}.json', dict(step=start, complete_state_verified=record['restore_verified']))
        del record
    else:
        save(out/'initial.pt', model, optimizer, scheduler, 0, config, provenance)
    with (out/f'rank{rank}.jsonl').open('a') as log:
        for step in range(start, stop):
            begun = time.monotonic(); episodes = []; targets = []; seeds = []
            for lane in range(batch):
                seed = plan.get('train_seed_start',34000000)+(step*world+rank)*batch+lane
                for attempt in range(100):
                    draw = seed+attempt*100000003
                    ep, target = factory.sample(draw, frames=1)
                    if not heldout(ep.stream): break
                else: raise RuntimeError('Unable to draw eligible training sequence')
                episodes.append(ep); targets.append(target); seeds.append(draw)
            truth = torch.stack([t[0][0] for t in targets])
            diameter = truth.new_tensor([float(e.mesh['diameter']) for e in episodes])
            noise = torch.randn(batch, 6, device='cuda')*truth.new_tensor([.15]*3+[.035]*3)
            if step % 10 == 0: noise.zero_()
            base = update(truth, noise[:, :3], noise[:, 3:], diameter)
            if step % 4 == 3: base = torch.stack([e.initial for e in episodes])
            scenes = [prepare_scene(e.rgb[0], e.depth[0], base[i], e.mesh, e.k, e.times[0], e.stream, e.cad, factory.renderer, fast=True) for i,e in enumerate(episodes)]
            occlusions = []
            for e,s,seed in zip(episodes, scenes, seeds):
                rng = np.random.default_rng(seed+34001)
                kind = int(rng.integers(4))
                oid = factory.streams[e.stream.split('|')[0]]['object_id']
                plan_occ = factory.occluders.plan(rng, oid, heavy=kind >= 2, start=0 if kind else 1, duration=1)
                occlusions.append(plan_occ.render(s, 0))
            data_time = time.monotonic()-begun
            optimizer.zero_grad(set_to_none=True)
            observation = encode_scenes(model, scenes, occlusions=occlusions)
            masks = torch.stack([t[1][0] for t in targets])
            target = build_teachers(model.ema_teacher, scenes, truth, masks, [o.mask for o in occlusions], factory.renderer,
                                    real_geometry_max_radius_d=1., fast=True, batch_render=True, vectorized=True, geometry_only=True)
            visible = torch.cat([(crop_images_fast(masks[i:i+1].float(), s.affine, mode='nearest') > .5) & ~o.mask & s.bounds for i,(s,o) in enumerate(zip(scenes,occlusions))])
            with torch.autocast('cuda', dtype=torch.bfloat16): output, _ = model(observation)
            loss, metrics = objective(output, target, observation, visible, torch.stack([s.k_crop for s in scenes]), config['cad_transport']['enabled'],
                                      canonical_surface=plan.get('canonical_surface',False),flow_weight=plan.get('flow_weight',.25))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite geometry loss')
            if step == 0 and not plan.get('decoder_only'):
                grad, = torch.autograd.grad(loss, output['patch_latent'], retain_graph=True)
                if not torch.isfinite(grad).all() or not grad.norm(): raise RuntimeError('Geometry does not train JEPA')
            loss.backward()
            if step == 0:
                names = () if plan.get('decoder_only') else ('core.src_proj.weight', 'surface_head.output.6.weight')
                if config['cad_transport']['enabled']: names += ('cad_transport.head.2.weight',)
                gradients = {n: float(p.grad.float().norm()) if p.grad is not None else None for n,p in model.named_parameters() if n in names}
                if len(gradients) != len(names) or any(v is None or v <= 0 for v in gradients.values()):
                    raise RuntimeError('Geometry path gradient failure: '+str(gradients))
                atomic_json(out/f'gradient_rank{rank}.json', gradients)
            synchronize_gradients(model.parameters())
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step()
            if not plan.get('decoder_only'): update_ema(model)
            row = dict(step=step+1, seconds=time.monotonic()-begun, data_seconds=data_time,
                       loss=float(loss.detach()), grad_norm=float(norm), lr={g['category']:g['lr'] for g in optimizer.param_groups},
                       metrics={k:float(v) for k,v in metrics.items()}, windows=[e.training_window for e in episodes])
            log.write(json.dumps(row)+'\n'); log.flush()
            if rank == 0 and (step+1)%25 == 0: print(json.dumps(row), flush=True)
            if (step+1)%50 == 0 or step+1 == stop:
                save(out/'last.pt', model, optimizer, scheduler, step+1, config, provenance)
            del output, observation, target, loss
    if rank == 0: atomic_json(out/'training_status.json', dict(step=stop, target=maximum, completed=stop==maximum, source_step=plan['source_step'], default_model_changed=False))
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
