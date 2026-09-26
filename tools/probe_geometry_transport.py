"""Fixed training-partition physical holdout. No validation/test tuning."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from prepare_serial_completion import heldout
from lip.unified.build import build_model, make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene, encode_scenes, build_teachers
from lip.unified.cad_transport import transport_targets, sample_reference, pixel_grid
from lip.unified.recovery_focus import camera_disagreement
from lip.unified.surface_normals import normal_diagnostics
from lip.engine.jepa_checkpoint import load_core, sha
from lip.geometry.so3 import update


def main():
    p = argparse.ArgumentParser()
    for key in ('config','checkpoint','out'): p.add_argument('--'+key, required=True)
    p.add_argument('--rank', type=int, default=0); p.add_argument('--world', type=int, default=8)
    p.add_argument('--records', type=int, default=8)
    p.add_argument('--lookup-audit', action='store_true')
    a = p.parse_args(); torch.cuda.set_device(0); torch.set_num_threads(2); torch.manual_seed(42)
    c = yaml.safe_load(Path(a.config).read_text()); model = build_model(c)
    record = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    load_core(model, record['model']); del record
    model.requires_grad_(False).eval(); model.fast_geometry = model.vector_geometry = True
    factory = Factory(c, model, make_store(c, model))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    rows = []; draw = 0; start = time.monotonic()
    with torch.no_grad(), (out/'frames.jsonl').open('w') as log:
        while len(rows) < a.records:
            seed = 44000000+a.rank+draw*a.world; draw += 1
            e, (truth, masks) = factory.sample(seed, frames=1)
            if not heldout(e.stream): continue
            item = len(rows); d = float(e.mesh['diameter']); gt = truth[:1]
            noise = gt.new_zeros(1, 6); noise[0, item%3] = (-1 if item%2 else 1)*torch.pi/18
            base = update(gt, noise[:, :3], noise[:, 3:], gt.new_tensor([d]))
            scene = prepare_scene(e.rgb[0], e.depth[0], base[0], e.mesh, e.k, e.times[0], e.stream, e.cad, factory.renderer, fast=True)
            oid = factory.streams[e.stream.split('|')[0]]['object_id']
            heavy = item%4 >= 2
            plan = factory.occluders.plan(np.random.default_rng(seed+341), oid, heavy=heavy, start=1 if item%4==0 else 0, duration=1,
                                         target_fraction=.8 if heavy else .3)
            occ = plan.render(scene, 0)
            obs = encode_scenes(model, [scene], occlusions=[occ])
            target = build_teachers(model.ema_teacher, [scene], gt, masks, [occ.mask], factory.renderer, real_geometry_max_radius_d=1.,
                                    fast=True, batch_render=True, vectorized=True, geometry_only=True)
            with torch.autocast('cuda', dtype=torch.bfloat16): output, _ = model(obs)
            truth_transport = transport_targets(target.surface_xyz, obs.geometry_image, obs.base, obs.diameter, scene.k_crop[None], obs.cad_valid)
            audit = {}
            if a.lookup_audit:
                bounds = obs.cad_valid.reshape(1,1,16,16).repeat_interleave(14,-2).repeat_interleave(14,-1)
                ref_mask = (obs.geometry_image[:,3:4] > 0) & bounds
                reference = torch.cat((obs.geometry_image[:,4:7],obs.geometry_image[:,2:3]),1)
                grid = pixel_grid(1,224,224,'cuda')
                moved,_ = sample_reference(reference,ref_mask,grid+output['transport_flow'])
                unmoved,mass = sample_reference(reference,ref_mask,grid)
                correction = output['transport_surface']-moved
                gate = output['transport_gate_logits'].sigmoid()*(mass >= .999)
                if not c['cad_transport']['enabled']: gate = gate*0.
                fallback = output['transport_fallback'][:,:4]
                zero_flow = fallback+gate*(unmoved+correction-fallback)
                gt_render = factory.renderer(scene.cad['appearance'],gt[0],scene.k_crop,224)
                gt_cad_depth = gt_render['depth'][None]
                audit = dict(zero_flow=zero_flow,reference=reference,reference_mask=ref_mask,gt_cad_depth=gt_cad_depth)
            def score(xyz, depth, mask):
                if not mask.any(): return None
                consistency = camera_disagreement(dict(surface_xyz=xyz, surface_depth_residual=depth), target).norm(dim=1, keepdim=True)
                error = (xyz-target.surface_xyz).norm(dim=1, keepdim=True)
                dep = (depth-target.surface_depth_residual).abs()
                return dict(pixels=int(mask.sum()), xyz_mm=float(error[mask].mean()*d*1000), depth_mm=float(dep[mask].mean()*d*1000),
                            xyz_median_mm=float(error[mask].median()*d*1000), xyz_p90_mm=float(error[mask].quantile(.9)*d*1000),
                            consistency_mm=float(consistency[mask].mean()*d*1000), xyz_within_10mm=float((error[mask]*d < .01).float().mean()))
            metrics = {}
            for name, mask in [('real', target.geometry_real_weight), ('proxy', target.geometry_proxy_weight)]:
                metrics[name] = score(output['surface_xyz'], output['surface_depth_residual'], mask)
                fallback = output['transport_fallback']
                metrics[name+'_fallback'] = score(fallback[:, :3], fallback[:, 3:4], mask)
                eligible = mask & truth_transport['supported']
                metrics[name+'_oracle_lookup'] = score(truth_transport['reference'][:, :3], target.surface_depth_residual, eligible)
                if metrics[name]:
                    metrics[name]['lookup_coverage'] = float(eligible.sum()/mask.sum())
                    metrics[name]['validity_recall'] = float((output['geometry_valid_logits'].sigmoid()[mask] >= .5).float().mean())
                    metrics[name]['normal'] = normal_diagnostics(output, target, mask)
                    metrics[name]['gate_mean'] = float(output['transport_gate'][mask].mean())
                    metrics[name]['flow_epe'] = float((output['transport_flow']-truth_transport['flow']).norm(dim=1, keepdim=True)[eligible].mean()) if eligible.any() else None
                    metrics[name]['zero_flow_epe'] = float(truth_transport['flow'].norm(dim=1,keepdim=True)[eligible].mean()) if eligible.any() else None
                if audit:
                    metrics[name+'_zero_flow'] = score(audit['zero_flow'][:,:3],audit['zero_flow'][:,3:4],mask)
                    overlap = mask & audit['reference_mask']
                    metrics[name+'_raw_cad_overlap'] = score(audit['reference'][:,:3],audit['reference'][:,3:4],overlap)
                    metrics[name+'_predicted_overlap'] = score(output['surface_xyz'],output['surface_depth_residual'],overlap)
                    if metrics[name]:
                        gap = (target.surface_depth_m-audit['gt_cad_depth']).abs()[mask]*1000
                        metrics[name]['real_to_gt_cad_depth_gap_mm'] = float(gap.mean())
                        metrics[name]['real_to_gt_cad_depth_gap_p90_mm'] = float(gap.quantile(.9))
            row = dict(seed=seed, stream=e.stream, heavy=heavy, natural=item%4==0, window=e.training_window, metrics=metrics)
            if item == 2:
                np.savez_compressed(out/'heavy_example.npz', rgb=scene.rgb.cpu().numpy(), occluded_rgb=occ.rgb.cpu().numpy(),
                    target_xyz=target.surface_xyz.cpu().numpy(), predicted_xyz=output['surface_xyz'].cpu().numpy(),
                    target_depth=target.surface_depth_m.cpu().numpy(), predicted_depth=output['surface_depth_m'].cpu().numpy(),
                    real_mask=target.geometry_real_weight.cpu().numpy(), proxy_mask=target.geometry_proxy_weight.cpu().numpy(),
                    flow=output['transport_flow'].cpu().numpy(), gate=output['transport_gate'].cpu().numpy(), diameter=d)
            rows.append(row); log.write(json.dumps(row)+'\n'); log.flush()
    (out/'receipt.json').write_text(json.dumps(dict(completed=True, checkpoint_sha256=sha(a.checkpoint), config=c['cad_transport'],
        records=len(rows), lookup_audit=a.lookup_audit, physical_holdout=True, training_split_only=True, official_test_access=False, teacher_inputs=False,
        oracle_lookup_scope='GT correspondence and GT depth diagnostic only; common supported pixels; never deployed', seconds=time.monotonic()-start), indent=2)+'\n')


if __name__ == '__main__': main()
