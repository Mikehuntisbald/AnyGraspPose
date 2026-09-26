"""Frozen causal write audit. Oracle endpoints are explicitly diagnostic only."""
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
            seed=61050000+a.rank+8*draw;draw+=1;e,(truth,masks)=f.sample(seed,frames=1)
            if not heldout(e.stream):continue
            item=len(rows);d=float(e.mesh['diameter']);gt=truth[:1];angle=10 if item%8<4 else 60
            noise=gt.new_zeros(1,6);noise[0,item%3]=angle*torch.pi/180*(-1 if item%2 else 1)
            base=update(gt,noise[:,:3],noise[:,3:],gt.new_tensor([d]))
            scene=prepare_scene(e.rgb[0],e.depth[0],base[0],e.mesh,e.k,e.times[0],e.stream,e.cad,f.renderer,fast=True)
            heavy=item%4>=2;oid=f.streams[e.stream.split('|')[0]]['object_id']
            plan=f.occluders.plan(np.random.default_rng(seed+581),oid,heavy=heavy,start=1 if item%4==0 else 0,duration=1,target_fraction=.8 if heavy else .3)
            occ=plan.render(scene,0);obs=encode_scenes(m,[scene],occlusions=[occ])
            with torch.autocast('cuda',dtype=torch.bfloat16):baseline,_=m(obs)
            target=build_teachers(m.ema_teacher,[scene],gt,masks,[occ.mask],f.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
            visible=(crop_images_fast(masks.float(),scene.affine,mode='nearest')>.5)&scene.bounds&~occ.mask
            labels=flow_labels(baseline,target,scene.k_crop[None],gt.new_tensor([d]),visible)
            known=labels['support']&baseline['flow_reference']['available']
            row=dict(seed=seed,stream=e.stream,angle=angle,heavy=heavy,correctable_points=int(known.sum()),variants={})
            for name,gain,oracle in [('normal',1,False),('off',0,False),('gain10',10,False),('gain100',100,False),('oracle',1,True),('oracle_gain10',10,True)]:
                hooks=[]
                def hook(module,args,output):
                    patch=args[0];updated,result=output
                    assert torch.equal(args[5]['uv'],baseline['flow_reference']['uv'])
                    uv=torch.where(known[...,None],labels['uv'],result['uv']) if oracle else result['uv']
                    write=rewrite(module,args,result,uv)
                    if not oracle:torch.testing.assert_close(write,result['write'],rtol=0,atol=0)
                    changed=patch+(gain*write).to(patch.dtype)
                    effective=changed.float()-patch.float()
                    hooks.append(dict(patch_dtype=str(patch.dtype),relative_write=float(write.norm()/patch.float().norm().clamp_min(1e-9)),
                        changed_fraction=float((changed!=patch).float().mean()),effective_relative_write=float(effective.norm()/patch.float().norm().clamp_min(1e-9))))
                    return changed,dict(result,uv=uv,flow=uv-args[5]['uv'],write=gain*write)
                h=m.flow_reconstruction.register_forward_hook(hook)
                try:
                    with torch.autocast('cuda',dtype=torch.bfloat16):out,_=m(obs)
                finally:h.remove()
                if name=='normal':
                    assert torch.equal(out['surface_xyz'],baseline['surface_xyz'])
                    assert torch.equal(out['surface_depth_residual'],baseline['surface_depth_residual'])
                result=dict(hooks=hooks,regions={})
                for region,mask in [('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight)]:
                    if not mask.any():continue
                    canonical=mask&target.cad_geometry_valid
                    result['regions'][region]=dict(pixels=int(mask.sum()),
                        canonical_xyz_mm=float((out['surface_xyz']-target.cad_geometry_xyz).norm(dim=1,keepdim=True)[canonical].mean()*d*1000) if canonical.any() else None,
                        depth_mm=float((out['surface_depth_residual']-target.surface_depth_residual).abs()[mask].mean()*d*1000),
                        xyz_change_mm=float((out['surface_xyz']-baseline['surface_xyz']).norm(dim=1,keepdim=True)[mask].mean()*d*1000),
                        depth_change_mm=float((out['surface_depth_residual']-baseline['surface_depth_residual']).abs()[mask].mean()*d*1000))
                row['variants'][name]=result
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
    (a.out/'receipt.json').write_text(json.dumps(dict(completed=True,records=len(rows),source_sha256=DIGEST,
        model_frozen=True,physical_holdout=True,training_partition_only=True,normal_rewrite_bitwise_verified=True,
        oracle_scope='GT endpoints replace only known front supported points; remaining endpoints/predicted support unchanged. Diagnostic only, not deployable.',
        seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':main()
