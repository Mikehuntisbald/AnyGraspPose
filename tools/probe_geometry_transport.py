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
    p.add_argument('--seed-start', type=int, default=44000000)
    p.add_argument('--lookup-audit', action='store_true')
    p.add_argument('--projection-audit', action='store_true')
    p.add_argument('--lip-baseline', action='store_true')
    p.add_argument('--clean-control', action='store_true',help='Remove synthetic input occlusion, but preserve original corrupted-case targets/masks')
    p.add_argument('--atlas-ablation',choices=['prior_only','learned_only'])
    p.add_argument('--correspondence-audit',action='store_true')
    p.add_argument('--recovery-routing',choices=['on','observed_only','rope_off'],default='on')
    p.add_argument('--flow-ablation', choices=['on','no_feedback','no_transport'], default='on')
    p.add_argument('--rotation-degrees', type=float, default=10.)
    p.add_argument('--flow-surface-audit', action='store_true')
    a = p.parse_args(); torch.cuda.set_device(0); torch.set_num_threads(2); torch.manual_seed(42)
    c = yaml.safe_load(Path(a.config).read_text()); model = build_model(c)
    record = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    load_core(model, record['model']); del record
    model.requires_grad_(False).eval(); model.fast_geometry = model.vector_geometry = True
    if a.flow_ablation != 'on':
        if not hasattr(model, 'flow_reconstruction'):raise ValueError('Flow ablation requires flow/recovery model')
        model.flow_reconstruction.disable_recovery_feedback = a.flow_ablation == 'no_feedback'
        model.flow_reconstruction.disable_transport = a.flow_ablation == 'no_transport'
    if a.recovery_routing=='observed_only':
        model.staged_rope['recovered_max_trust']=0.
    elif a.recovery_routing=='rope_off':
        with torch.no_grad():model.cad_surface.rope3d.gain.zero_()
    atlas=hasattr(model,'cad_atlas_decoder')
    if a.atlas_ablation:
        if not atlas:raise ValueError('Atlas ablation requires the atlas decoder')
        atlas_scores=model.cad_atlas_decoder.scores
        if a.atlas_ablation=='prior_only':
            model.cad_atlas_decoder.scores=lambda q,k,p,x,v,fn=atlas_scores:fn(torch.zeros_like(q),k,p,x,v)
        else:
            model.cad_atlas_decoder.scores=lambda q,k,p,x,v,fn=atlas_scores:fn(q,k,torch.zeros_like(p),torch.zeros_like(x),v)
    factory = Factory(c, model, make_store(c, model))
    lip=None;lip_sha=None
    if a.lip_baseline:
        from lip.models.stream_tracker import StreamTracker
        from lip.engine.stream_features import build_current_features,stack_current
        from lip.engine.stream_state import FrameMeta
        from lip.geometry.so3 import log as rotation_log
        path=Path('/mnt/why/dexycb_lip/smooth_val_evaluation_20260915/runs/full/selected.pt')
        lip_sha=sha(path)
        assert lip_sha=='89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868'
        saved=torch.load(path,map_location='cpu',weights_only=False);lc=saved['config']
        lip=StreamTracker('stream_dual_cross_residual',lc['memory_frames'],lc.get('dropout',0.),False,
                          lc.get('time_unit',1/30),lc.get('max_gap_seconds',.5),'functional').cuda()
        lip.load_state_dict(saved['model'],strict=True);del saved
        lip.requires_grad_(False).eval()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    rows = []; draw = 0; start = time.monotonic()
    with torch.no_grad(), (out/'frames.jsonl').open('w') as log:
        while len(rows) < a.records:
            seed = a.seed_start+a.rank+draw*a.world; draw += 1
            e, (truth, masks) = factory.sample(seed, frames=1)
            if not heldout(e.stream): continue
            item = len(rows); d = float(e.mesh['diameter']); gt = truth[:1]
            noise = gt.new_zeros(1, 6); noise[0, item%3] = (-1 if item%2 else 1)*torch.pi*a.rotation_degrees/180
            base = update(gt, noise[:, :3], noise[:, 3:], gt.new_tensor([d]))
            scene = prepare_scene(e.rgb[0], e.depth[0], base[0], e.mesh, e.k, e.times[0], e.stream, e.cad, factory.renderer, fast=True)
            oid = factory.streams[e.stream.split('|')[0]]['object_id']
            heavy = item%4 >= 2
            plan = factory.occluders.plan(np.random.default_rng(seed+341), oid, heavy=heavy, start=1 if item%4==0 else 0, duration=1,
                                         target_fraction=.8 if heavy else .3)
            occ = plan.render(scene, 0)
            obs = encode_scenes(model, [scene], occlusions=[None] if a.clean_control else [occ])
            target = build_teachers(model.ema_teacher, [scene], gt, masks, [occ.mask], factory.renderer, real_geometry_max_radius_d=1.,
                                    fast=True, batch_render=True, vectorized=True, geometry_only=True)
            with torch.autocast('cuda', dtype=torch.bfloat16): output, _ = model(obs)
            lip_result=None
            if lip is not None:
                features,diag=build_current_features(e.rgb[0],e.depth[0],base[0],e.k,e.mesh,factory.renderer.geometry,
                    e.times[0],e.times[0]-1/30,size=224,expansion=2.)
                torch.testing.assert_close(diag['A'],scene.affine,atol=2e-5,rtol=0.)
                torch.testing.assert_close(diag['K_crop'],scene.k_crop,atol=2e-5,rtol=0.)
                # Exact same corrupted RGB/geometry tensors as the JEPA student.
                features['rgb']=obs.packet.rgb_crop[0]
                features['geometry']=obs.geometry_image[0]
                features['state_input'][-1]=(obs.measured_depth_m>0).float().mean()
                meta=FrameMeta(torch.tensor([e.times[0]],device='cuda',dtype=torch.float64),torch.ones(1,device='cuda',dtype=torch.long),
                               torch.zeros(1,device='cuda',dtype=torch.long),torch.ones(1,17,device='cuda',dtype=torch.bool),diag['role_bias'][None])
                with torch.autocast('cuda',dtype=torch.bfloat16):result,_=lip(stack_current([features]),meta,None)
                pose=result['pose_centered'][0]
                if not torch.isfinite(pose).all():raise RuntimeError('Nonfinite frozen LIP output')
                render=factory.renderer(scene.cad['appearance'],pose,scene.k_crop,224)
                lip_result=dict(render=render,rotation_before_deg=float(rotation_log(base[0,:3,:3]@gt[0,:3,:3].T).norm()*180/torch.pi),
                    rotation_after_deg=float(rotation_log(pose[:3,:3]@gt[0,:3,:3].T).norm()*180/torch.pi),
                    translation_after_mm=float((pose[:3,3]-gt[0,:3,3]).norm()*1000))
            truth_transport = transport_targets(target.surface_xyz, obs.geometry_image, obs.base, obs.diameter, scene.k_crop[None], obs.cad_valid)
            cad_truth = transport_targets(target.cad_geometry_xyz, obs.geometry_image, obs.base, obs.diameter, scene.k_crop[None], obs.cad_valid)
            audit = {}
            if a.lookup_audit and not atlas:
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
            projection = {}
            if a.projection_audit and not atlas:
                # All inputs here are student predictions or estimated-pose CAD.
                # The true transform and teacher masks are used only in score().
                fallback = output['atlas_fallback'] if atlas else output['transport_fallback']
                camera = torch.einsum('bij,bjhw->bihw',obs.base[:,:3,:3],fallback[:,:3])*d+obs.base[:,:3,3,None,None]
                homogeneous = torch.einsum('bij,bjhw->bihw',scene.k_crop[None],camera)
                uv = homogeneous[:,:2]/homogeneous[:,2:3].clamp_min(.001)
                bounds = obs.cad_valid.reshape(1,1,16,16).repeat_interleave(14,-2).repeat_interleave(14,-1)
                mask = (obs.geometry_image[:,3:4] > 0) & bounds
                reference = torch.cat((obs.geometry_image[:,4:7],obs.geometry_image[:,2:3]),1)
                snapped,mass = sample_reference(reference,mask,uv)
                available = (mass >= .999) & (camera[:,2:3] > .001)
                surface = torch.where(available,snapped,fallback[:,:4])
                projection = dict(surface=surface,available=available,flow=uv-pixel_grid(1,224,224,'cuda'))
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
                fallback = output['atlas_fallback'] if atlas else output['transport_fallback']
                metrics[name+'_fallback'] = score(fallback[:, :3], fallback[:, 3:4], mask)
                eligible = mask & truth_transport['supported']
                metrics[name+'_oracle_lookup'] = score(truth_transport['reference'][:, :3], target.surface_depth_residual, eligible)
                if metrics[name]:
                    cad_domain=mask&target.cad_geometry_valid
                    cad_eligible=cad_domain&cad_truth['supported']
                    metrics[name]['canonical_pixels']=int(cad_domain.sum())
                    metrics[name]['canonical_xyz_mm']=float((output['surface_xyz']-target.cad_geometry_xyz).norm(dim=1,keepdim=True)[cad_domain].mean()*d*1000) if cad_domain.any() else None
                    metrics[name]['cad_flow_epe']=float((output['transport_flow']-cad_truth['flow']).norm(dim=1,keepdim=True)[cad_eligible].mean()) if cad_eligible.any() and not atlas else None
                    metrics[name]['cad_zero_flow_epe']=float(cad_truth['flow'].norm(dim=1,keepdim=True)[cad_eligible].mean()) if cad_eligible.any() else None
                    metrics[name]['cad_lookup_coverage']=float(cad_eligible.sum()/mask.sum())
                    metrics[name]['lookup_coverage'] = float(eligible.sum()/mask.sum())
                    metrics[name]['validity_recall'] = float((output['geometry_valid_logits'].sigmoid()[mask] >= .5).float().mean())
                    metrics[name]['normal'] = normal_diagnostics(output, target, mask)
                    # Physical camera geometry is distinct from canonical CAD identity.
                    camera_pred=target.camera_rays*output['surface_depth_m']
                    camera_target=target.camera_rays*target.surface_depth_m
                    metrics[name]['camera_xyz_mm']=float((camera_pred-camera_target).norm(dim=1,keepdim=True)[mask].mean()*1000)
                    from lip.unified.surface_normals import normal_terms
                    _,camera_valid,camera_cos=normal_terms(camera_pred/d,camera_target/d,mask,2)
                    metrics[name]['camera_normal_deg']=float(camera_cos[camera_valid].acos().mean()*180/torch.pi) if camera_valid.any() else None
                    metrics[name]['camera_normal_stencils']=int(camera_valid.sum())
                    metrics[name]['gate_mean'] = float(output['transport_gate'][mask].mean()) if not atlas else None
                    metrics[name]['flow_epe'] = float((output['transport_flow']-truth_transport['flow']).norm(dim=1, keepdim=True)[eligible].mean()) if eligible.any() and not atlas else None
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
                if projection:
                    projected = projection['surface']
                    metrics[name+'_projected_xyz'] = score(projected[:,:3],output['transport_fallback'][:,3:4],mask)
                    metrics[name+'_projected_xyz_depth'] = score(projected[:,:3],projected[:,3:4],mask)
                    if metrics[name]:
                        metrics[name]['predicted_projection_coverage'] = float(projection['available'][mask].float().mean())
                        metrics[name]['predicted_projection_epe'] = float((projection['flow']-truth_transport['flow']).norm(dim=1,keepdim=True)[eligible].mean()) if eligible.any() else None
            row = dict(seed=seed, stream=e.stream, heavy=heavy, natural=item%4==0, clean_control=a.clean_control,window=e.training_window, metrics=metrics)
            if 'flow_rounds' in output:
                from lip.unified.flow_reconstruction import flow_labels, flow_reconstruction_loss
                from lip.unified.execution_speed import crop_images_fast
                flow_visible = (crop_images_fast(masks[:1].float(), scene.affine, mode='nearest') > .5) & scene.bounds
                if not a.clean_control:flow_visible = flow_visible & ~occ.mask
                labels = flow_labels(output, target, scene.k_crop[None], obs.diameter, flow_visible)
                _, values = flow_reconstruction_loss(output, labels)
                row['flow'] = {k:float(v) for k,v in values.items()}
                row['flow_ablation'] = a.flow_ablation
                if a.flow_surface_audit:
                    from lip.unified.flow_surface_audit import diagnose_flow_surface
                    row['flow_surface'], surfaces = diagnose_flow_surface(output, target, scene, obs, flow_visible, factory.renderer, seed)
                    if item == 2:
                        np.savez_compressed(out/'surface_audit_example.npz',
                            **{f'{name}_{key}':value[key].cpu().numpy() for name,value in surfaces.items() for key in ('xyz','depth','covered')})
                if item == 2:
                    np.savez_compressed(out/'flow_example.npz', reference_uv=output['flow_reference']['uv'].cpu().numpy(),
                        reference_xyz=output['flow_reference']['xyz'].cpu().numpy(), target_uv=labels['uv'].cpu().numpy(),
                        uv0=output['flow_rounds'][0]['uv'].cpu().numpy(), uv1=output['flow_rounds'][1]['uv'].cpu().numpy(),
                        **{name:labels[name].cpu().numpy() for name in ('observed','real','proxy')})
            if 'cad_image_uv' in output or a.correspondence_audit:
                from cad_image_diagnostics import diagnose
                from lip.unified.execution_speed import crop_images_fast
                visible=(crop_images_fast(masks[:1].float(),scene.affine,mode='nearest')>.5)&scene.bounds
                if not a.clean_control:visible=visible&~occ.mask
                row['correspondence']=diagnose(output,target,scene,obs,visible,gt[0],e.mesh,seed,factory.renderer if a.correspondence_audit else None,model if a.correspondence_audit else None)
                if item==2 and 'cad_image_uv' in output:
                    from lip.unified.cad_image_correspondence import image_correspondence_targets
                    endpoint_labels=image_correspondence_targets(output,target,scene.k_crop[None],obs.diameter,visible)
                    uv=output['cad_image_uv'][0]
                    native=torch.cat((uv,torch.ones_like(uv[:,:1])),1)@torch.linalg.inv(scene.affine).T
                    np.savez_compressed(out/'correspondences.npz',cad_ids=output['cad_image_ids'].cpu().numpy(),
                        cad_xyz_m=output['cad_image_xyz'][0].cpu().numpy()*d,uv_crop=uv.cpu().numpy(),
                        uv_image=(native[:,:2]/native[:,2:]).cpu().numpy(),confidence=output['cad_image_confidence'][0].cpu().numpy(),
                        visibility=output['cad_image_visible_logits'][0].sigmoid().cpu().numpy(),
                        support=output['cad_image_support_logits'][0].sigmoid().cpu().numpy(),
                        gt_uv_crop=endpoint_labels['uv'][0].cpu().numpy(),gt_support=endpoint_labels['support'][0].cpu().numpy(),
                        gt_visible=endpoint_labels['visible'][0].cpu().numpy(),
                        **(dict(flow_uv_crop=output['cad_flow_uv'][0].cpu().numpy(),flow_reference=output['cad_flow_reference'][0].cpu().numpy()) if 'cad_flow_uv' in output else {}))
            if lip_result is not None:
                row['lip_pose']={k:v for k,v in lip_result.items() if k!='render'}
                row['geometry_baselines']={}
                for tag,xyz,depth,valid in [('base_cad',scene.render['xyz'][None]/d,scene.render['depth'][None],scene.render['mask'][None,None]),
                    ('lip_cad',lip_result['render']['xyz'][None]/d,lip_result['render']['depth'][None],lip_result['render']['mask'][None,None]),
                    ('jepa',output['surface_xyz'],output['surface_depth_m'],output['geometry_valid_logits'].sigmoid()>=.5)]:
                    row['geometry_baselines'][tag]={}
                    error=(xyz-target.cad_geometry_xyz).norm(dim=1,keepdim=True)*d*1000
                    depth_error=(depth-target.surface_depth_m).abs()*1000
                    for kind,mask in [('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight)]:
                        if not mask.any():row['geometry_baselines'][tag][kind]=None;continue
                        covered=mask&valid
                        row['geometry_baselines'][tag][kind]=dict(pixels=int(mask.sum()),covered_fraction=float(covered.sum()/mask.sum()),
                            covered_canonical_xyz_mm=float(error[covered].mean()) if covered.any() else None,
                            covered_depth_mm=float(depth_error[covered].mean()) if covered.any() else None,
                            all_target_good_10mm_xyz_5mm_depth=float((valid&(error<10)&(depth_error<5)&mask).sum()/mask.sum()))
            if item == 2:
                np.savez_compressed(out/'heavy_example.npz', rgb=scene.rgb.cpu().numpy(), occluded_rgb=(scene.rgb if a.clean_control else occ.rgb).cpu().numpy(),
                    target_xyz=target.surface_xyz.cpu().numpy(), predicted_xyz=output['surface_xyz'].cpu().numpy(),
                    cad_target_xyz=target.cad_geometry_xyz.cpu().numpy(),cad_target_depth=target.cad_geometry_depth_m.cpu().numpy(),
                    target_depth=target.surface_depth_m.cpu().numpy(), predicted_depth=output['surface_depth_m'].cpu().numpy(),
                    real_mask=target.geometry_real_weight.cpu().numpy(), proxy_mask=target.geometry_proxy_weight.cpu().numpy(),
                    diameter=d,**(dict(atlas_index=output['atlas_index'].cpu().numpy()) if atlas else dict(flow=output['transport_flow'].cpu().numpy(),gate=output['transport_gate'].cpu().numpy())))
            rows.append(row); log.write(json.dumps(row)+'\n'); log.flush()
    (out/'receipt.json').write_text(json.dumps(dict(completed=True, checkpoint_sha256=sha(a.checkpoint), config=c['cad_transport'],
        records=len(rows), seed_start=a.seed_start, lookup_audit=a.lookup_audit, projection_audit=a.projection_audit, physical_holdout=True, training_split_only=True, official_test_access=False, teacher_inputs=False,
        clean_control=a.clean_control,clean_control_scope='Artificial occlusion removed from current input only; original target masks retained. Privileged input-availability diagnostic, never a heavy-occlusion deployment result' if a.clean_control else None,
        atlas_ablation=a.atlas_ablation,
        correspondence_audit=a.correspondence_audit,
        recovery_routing=a.recovery_routing,
        flow_ablation=a.flow_ablation,
        rotation_degrees=a.rotation_degrees,
        flow_surface_audit=a.flow_surface_audit,
        lip_baseline_sha256=lip_sha,lip_baseline_scope='Same corrupted current input/base/crop, empty history, one refinement; no GT render fed to LIP; conditional diagnostic, not native LIP accuracy' if lip is not None else None,
        oracle_lookup_scope='GT correspondence and GT depth diagnostic only; common supported pixels; never deployed', seconds=time.monotonic()-start), indent=2)+'\n')


if __name__ == '__main__': main()
