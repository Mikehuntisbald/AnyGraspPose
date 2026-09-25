"""Frozen V21 objective/label audit and frozen-JEPA geometry-decoder cache.

Train data only. No production optimizer updates. The saved decoder inputs are
detached and cannot expose teacher targets to the student encoder.
"""
import argparse,json,sys,time
from pathlib import Path
import torch,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.unified.execution_speed import crop_images_fast
from lip.unified.cad_surface_targets import surface_targets
from lip.unified.serial_objective import objective,delta_target
from lip.unified.recovery_focus import camera_disagreement
from lip.unified.losses import reconstruction_loss
from lip.unified.local_structure import local_structure_loss
from lip.engine.jepa_checkpoint import load_core,sha
from lip.geometry.so3 import update
from lip.losses import pose_loss


def compare(a,b,mask=None):
    a=a.detach().float();b=b.detach().float()
    if mask is not None:a=a*mask;b=b*mask
    aa=float(a.norm());bb=float(b.norm());dot=float((a*b).sum())
    return dict(pose_norm=aa,geometry_norm=bb,cosine=dot/(aa*bb) if min(aa,bb)>1e-8 else None,
                pose_to_geometry_norm=aa/bb if bb>1e-8 else None,dot=dot)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8)
    p.add_argument('--records',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(42)
    c=yaml.safe_load(Path(a.config).read_text());m=build_model(c)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);load_core(m,saved['model']);del saved
    m.requires_grad_(False).eval();m.core.requires_grad_(True);m.surface_head.requires_grad_(True)
    m.fast_geometry=m.vector_geometry=True;factory=Factory(c,m,make_store(c,m))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rows=[];cache=[];started=time.monotonic()
    captured=[]
    hook=m.surface_head.register_forward_pre_hook(lambda module,args:captured.append(([x.detach() for x in args[0]],args[1].detach())))
    for item in range(a.records):
        seed=18000000+a.rank+item*a.world;e,(truth,visible)=factory.sample(seed,frames=12);frame=8
        # Fix severe textured occlusion so the audit actually samples restoration.
        e.occlusion_plan=factory.occluders.plan(__import__('numpy').random.default_rng(seed),int(e.cad.get('object_id',0)) if 'object_id' in e.cad else factory.streams[e.stream.split('|')[0]]['object_id'],heavy=True,start=4,duration=8,target_fraction=.8)
        d=float(e.mesh['diameter']);gt=truth[frame:frame+1]
        noise=gt.new_zeros(1,6);noise[0,item%3]=(-1 if item%2 else 1)*torch.pi/18
        base=update(gt,noise[:,:3],noise[:,3:],gt.new_tensor([d]))[0]
        scene=prepare_scene(e.rgb[frame],e.depth[frame],base,e.mesh,e.k,e.times[frame],e.stream,e.cad,factory.renderer,fast=True)
        occ=e.occlusion_plan.render(scene,frame);obs=encode_scenes(m,[scene],occlusions=[occ])
        target=build_teachers(m.ema_teacher,[scene],gt,visible[frame:frame+1],[occ.mask],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True)
        target=surface_targets(m,[scene],target)
        vm=(crop_images_fast(visible[frame:frame+1].float(),scene.affine,mode='nearest')>.5)&~occ.mask&scene.bounds
        captured.clear()
        with torch.autocast('cuda',dtype=torch.bfloat16):output,_=m(obs)
        points=torch.as_tensor(e.mesh['points'],device='cuda')[None]
        pose,_=pose_loss(output['pose_centered'],gt,points,obs.diameter)
        delta=torch.cat((output['delta_rotvec'],output['delta_center_norm']),-1)
        scale=delta.new_tensor([.174533]*3+[.05]*3)
        direct=F.smooth_l1_loss(delta/scale,delta_target(obs.base,gt,obs.diameter)/scale,beta=.1)
        pose=pose+.1*direct
        weights=c['training']['loss_weights']
        feature_weights=dict(weights,surface_xyz=0.,surface_depth=0.,geometry_validity=0.,visibility_support=0.)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            total,_=objective(output,target,gt,points,obs.diameter,obs.base,obs.measured_depth_m,weights,vm)
            appearance,_=reconstruction_loss(output,target,feature_weights)
            local,_=local_structure_loss(output,target,weights)
        geo=total-pose-appearance-local
        tensors=(output['surface_xyz'],output['surface_depth_residual'],output['patch_latent'])
        gp=torch.autograd.grad(pose,tensors,retain_graph=True)
        gg=torch.autograd.grad(geo,tensors)
        stats={name:compare(x,y) for name,x,y in zip(('xyz','depth','patch'),gp,gg)}
        for name,mask in (('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight),('visible',vm)):
            stats[name]={k:compare(x,y,mask) for k,x,y in zip(('xyz','depth'),gp[:2],gg[:2])}
        with torch.no_grad():
            ideal=camera_disagreement(dict(surface_xyz=target.surface_xyz,surface_depth_residual=target.surface_depth_residual),target).norm(dim=1,keepdim=True)*d*1000
            gt_render=factory.renderer(e.cad['appearance'],gt[0],scene.k_crop,224)
            rendered_depth=gt_render['depth'][None]
            gap=(target.surface_depth_m-rendered_depth).abs()*1000
            measured_xyz=torch.einsum('bji,bjhw->bihw',gt[:,:3,:3],target.camera_rays*obs.measured_depth_m/obs.diameter[:,None,None,None]-target.camera_translation_d[:,:,None,None])
            observed_valid=vm&(obs.measured_depth_m>0)&torch.isfinite(obs.measured_depth_m)&(measured_xyz.square().sum(1,keepdim=True)<1.)
            metrics={}
            for kind,mask in (('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight)):
                pixels=int(mask.sum());metrics[kind]=dict(pixels=pixels,
                    target_camera_consistency_mm=float(ideal[mask].mean()) if pixels else None,
                    sensor_render_depth_gap_mm=float(gap[mask].mean()) if pixels else None,
                    sensor_render_gap_gt50mm=float((gap[mask]>50).float().mean()) if pixels else None,
                    target_canonical_outside_radius_half=float((target.surface_xyz.norm(dim=1,keepdim=True)[mask]>.55).float().mean()) if pixels else None)
            metrics['feature_patches_real']=float(target.hidden_real_weight.sum());metrics['feature_patches_proxy']=float(target.proxy_weight.sum())
            metrics['pixels_real']=int(target.geometry_real_weight.sum());metrics['pixels_proxy']=int(target.geometry_proxy_weight.sum())
            row=dict(seed=seed,stream=e.stream,gradients=stats,targets=metrics,pose_loss=float(pose),geometry_loss=float(geo))
            rows.append(row)
            cpu=lambda x:x.detach().cpu()
            levels,valid=captured[-1]
            cache.append(dict(seed=seed,stream=e.stream,levels=[cpu(x) for x in levels],valid=cpu(valid),diameter=d,
                xyz=cpu(target.surface_xyz),depth=cpu(target.surface_depth_residual),validity=cpu(target.geometry_valid_label),
                real_mask=cpu(target.geometry_real_weight),proxy_mask=cpu(target.geometry_proxy_weight),
                observed_xyz=cpu(measured_xyz),observed_mask=cpu(observed_valid),rotation=cpu(gt[:,:3,:3]),
                translation_d=cpu(target.camera_translation_d),rays=cpu(target.camera_rays),base_depth_d=cpu(target.base_depth_d)))
        del output,pose,geo,total,gp,gg,tensors
        print(json.dumps(dict(rank=a.rank,records=len(rows),seconds=time.monotonic()-started)),flush=True)
    hook.remove();torch.save(cache,out/'decoder_cache.pt')
    (out/'gradients.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,checkpoint_sha256=sha(a.checkpoint),records=len(rows),optimizer_updates=0,
        train_split_only=True,official_test_access=False,decoder_cache_sha256=sha(out/'decoder_cache.pt'),seconds=time.monotonic()-started),indent=2)+'\n')

if __name__=='__main__':main()
