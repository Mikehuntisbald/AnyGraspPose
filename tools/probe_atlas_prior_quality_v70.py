"""Frozen flow-to-atlas canonical prior audit. No training or pose solver."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_serial_completion import heldout
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.unified.execution_speed import crop_images_fast
from lip.unified.flow_reconstruction import flow_labels,patch_grid
from lip.geometry.so3 import update
from lip.engine.jepa_checkpoint import load_core,sha
SOURCE='/mnt/why/dexycb_lip/unified_jepa_20260921/local_flow_v60_r1/training/extra.pt'
DIGEST='0c6f712c7848316259dae3d24554dd4dfac506a20404676e43fb679b94d46d7e'


@torch.autocast('cuda',enabled=False)
def rewrite(module,args,result,uv):
    patch,observed,cad,geometry,valid,reference,*_=args
    grid=patch_grid(patch.device)
    source=F.normalize(module.descriptor(cad.float()),dim=-1)
    weights=(-(uv[:,:,None]-grid[None,None]).square().sum(-1)/(2*14.**2)).exp()
    trust=(.1+.9*result['support_logits'].sigmoid())*reference['available']
    weights=weights*trust[...,None]*valid[:,None]
    mass=weights.sum(1)
    values=torch.cat((source,reference['xyz'],reference['depth']),-1)
    aligned=(weights.transpose(1,2)@values)/mass[...,None].clamp_min(1e-6)
    ov=geometry[:,1:2].float().clamp(0,1)
    om=F.avg_pool2d(ov,14).flatten(1)
    od=F.avg_pool2d(geometry[:,:1].float()*ov,14).flatten(1)/om.clamp_min(1e-6)
    residual=(aligned[...,-1]-od)*om
    context=torch.cat((aligned,mass[...,None].clamp_max(4),od[...,None],om[...,None],residual[...,None]),-1)
    return .1*module.write(context)*(mass/(mass+1))[...,None]*valid[...,None]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--records',type=int,default=8);a=p.parse_args()
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(42)
    assert sha(SOURCE)==DIGEST
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False);c=saved['config'];m=build_model(c);load_core(m,saved['model']);del saved
    m.requires_grad_(False).eval();m.fast_geometry=m.vector_geometry=True
    f=Factory(c,m,make_store(c,m));a.out.mkdir(exist_ok=False,parents=True)
    rows=[];draw=0;start=time.monotonic()
    with torch.no_grad(),(a.out/'frames.jsonl').open('w') as log:
        while len(rows)<a.records:
            seed=66050000+a.rank+8*draw;draw+=1;e,(truth,masks)=f.sample(seed,frames=1)
            if not heldout(e.stream):continue
            item=len(rows);d=float(e.mesh['diameter']);gt=truth[:1];angle=10 if item%8<4 else 60
            noise=gt.new_zeros(1,6);noise[0,item%3]=angle*torch.pi/180*(-1 if item%2 else 1)
            base=update(gt,noise[:,:3],noise[:,3:],gt.new_tensor([d]))
            scene=prepare_scene(e.rgb[0],e.depth[0],base[0],e.mesh,e.k,e.times[0],e.stream,e.cad,f.renderer,fast=True)
            heavy=item%4>=2;oid=f.streams[e.stream.split('|')[0]]['object_id']
            plan=f.occluders.plan(np.random.default_rng(seed+581),oid,heavy=heavy,start=1 if item%4==0 else 0,duration=1,target_fraction=.8 if heavy else .3)
            occ=plan.render(scene,0);obs=encode_scenes(m,[scene],occlusions=[occ])
            m.cad_atlas_decoder.prior_sigma=.1
            with torch.autocast('cuda',dtype=torch.bfloat16):baseline,_=m(obs)
            target=build_teachers(m.ema_teacher,[scene],gt,masks,[occ.mask],f.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
            visible=(crop_images_fast(masks.float(),scene.affine,mode='nearest')>.5)&scene.bounds&~occ.mask
            labels=flow_labels(baseline,target,scene.k_crop[None],gt.new_tensor([d]),visible)
            known=labels['support']&baseline['flow_reference']['available']
            row=dict(seed=seed,stream=e.stream,angle=angle,heavy=heavy,correctable_points=int(known.sum()),variants={})
            for name in ('normal','pred14','oracle14','oracle_supported14','pred14_tight','oracle14_tight','oracle_supported14_tight','perfect_prior','perfect_prior_tight'):
                hooks=[];captured={}
                m.cad_atlas_decoder.prior_sigma=.03 if name.endswith('_tight') else .1
                def prehook(module,args):
                    from lip.unified.flow_atlas_prior import aligned_canonical_prior
                    reference=baseline['flow_reference']
                    changed=args[1]
                    coverage=torch.zeros_like(target.cad_geometry_valid)
                    available=reference['available']
                    if name.startswith('perfect_prior'):
                        changed=changed.clone()
                        coverage=target.cad_geometry_valid
                        changed[:,:3]=torch.where(coverage,target.cad_geometry_xyz,changed[:,:3])
                    elif name!='normal':
                        uv=baseline['flow_rounds'][-1]['uv']
                        if name.startswith('oracle'):
                            uv=torch.where(known[...,None],labels['uv'],uv)
                        available=known if name.startswith('oracle_supported') else reference['available']
                        changed,coverage=aligned_canonical_prior(reference['xyz'],uv,available,args[1],radius=14.)
                    captured.update(prior=changed[:,:3],coverage=coverage)
                    hooks.append(dict(coverage_pixels=int(coverage.sum()),available_anchors=int(available.sum()),prior_sigma=module.prior_sigma))
                    return None if name=='normal' else (args[0],changed,*args[2:])
                h=m.cad_atlas_decoder.register_forward_pre_hook(prehook)
                try:
                    with torch.autocast('cuda',dtype=torch.bfloat16):out,_=m(obs)
                finally:h.remove()
                if name=='normal':
                    assert torch.equal(out['surface_xyz'],baseline['surface_xyz'])
                    assert torch.equal(out['flow_rounds'][-1]['uv'],baseline['flow_rounds'][-1]['uv'])
                assert torch.equal(out['surface_depth_residual'],baseline['surface_depth_residual'])
                assert torch.equal(out['flow_rounds'][-1]['uv'],baseline['flow_rounds'][-1]['uv'])
                result=dict(hooks=hooks,regions={},flow={})
                for stage,flow in enumerate(out['flow_rounds']):
                    errors=(flow['uv']-labels['uv']).norm(dim=-1)
                    result['flow'][str(stage)]={region:dict(points=int(labels[region].sum()),epe=float(errors[labels[region]].mean()),
                        within3=float((errors[labels[region]]<=3).float().mean()))
                        for region in ('observed','real','proxy') if labels[region].any()}
                result['feedback_abs_mean']=float(out['flow_rounds'][-1]['geometry_feedback'].abs().mean())
                for region,mask in [('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight),('observed',visible)]:
                    if not mask.any():continue
                    canonical=mask&target.cad_geometry_valid
                    result['regions'][region]=dict(pixels=int(mask.sum()),
                        canonical_xyz_mm=float((out['surface_xyz']-target.cad_geometry_xyz).norm(dim=1,keepdim=True)[canonical].mean()*d*1000) if canonical.any() else None,
                        depth_mm=float((out['surface_depth_residual']-target.surface_depth_residual).abs()[mask].mean()*d*1000),
                        xyz_change_mm=float((out['surface_xyz']-baseline['surface_xyz']).norm(dim=1,keepdim=True)[mask].mean()*d*1000),
                        depth_change_mm=float((out['surface_depth_residual']-baseline['surface_depth_residual']).abs()[mask].mean()*d*1000))
                    if canonical.any():
                        prior_error=(captured['prior']-target.cad_geometry_xyz).norm(dim=1,keepdim=True)*d*1000
                        final_error=(out['surface_xyz']-target.cad_geometry_xyz).norm(dim=1,keepdim=True)*d*1000
                        covered=canonical&captured['coverage'];uncovered=canonical&~captured['coverage']
                        def masked_mean(value,selected):
                            return float(value[selected].mean()) if selected.any() else None
                        result['regions'][region].update(canonical_pixels=int(canonical.sum()),covered_pixels=int(covered.sum()),
                            prior_xyz_mm=masked_mean(prior_error,canonical),
                            covered_prior_xyz_mm=masked_mean(prior_error,covered),covered_final_xyz_mm=masked_mean(final_error,covered),
                            uncovered_prior_xyz_mm=masked_mean(prior_error,uncovered),uncovered_final_xyz_mm=masked_mean(final_error,uncovered))
                row['variants'][name]=result
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
    (a.out/'receipt.json').write_text(json.dumps(dict(completed=True,records=len(rows),source_sha256=DIGEST,
        model_frozen=True,physical_holdout=True,training_partition_only=True,normal_forward_bitwise_verified=True,
        oracle_scope='Oracle flow variants use GT-supported endpoints; perfect_prior replaces the canonical prior on GT-valid pixels. Tight variants use sigma .03 instead of .1. All are frozen diagnostics; depth and flow unchanged. Covered metrics are secondary, primary metrics retain all targets.',
        seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':main()
