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
SOURCE='/mnt/why/dexycb_lip/unified_jepa_20260921/local_projection_joint_v72_fast/surface/seed42/last.pt'
DIGEST='9a5605822cf9a5407193735a7989fe435d8774e48b6e09a3fcb41cb1e1e7cbec'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--records',type=int,default=12);a=p.parse_args()
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(42)
    assert sha(SOURCE)==DIGEST
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False);c=saved['config'];m=build_model(c);load_core(m,saved['model']);del saved
    m.requires_grad_(False).eval();m.fast_geometry=m.vector_geometry=True
    f=Factory(c,m,make_store(c,m));a.out.mkdir(exist_ok=False,parents=True)
    rows=[];draw=0;start=time.monotonic()
    with torch.no_grad(),(a.out/'frames.jsonl').open('w') as log:
        while len(rows)<a.records:
            seed=73050000+a.rank+8*draw;draw+=1;e,(truth,masks)=f.sample(seed,frames=1)
            if not heldout(e.stream):continue
            item=len(rows);d=float(e.mesh['diameter']);gt=truth[:1];angle=(0,10,60)[item//4]
            noise=gt.new_zeros(1,6);noise[0,item%3]=angle*torch.pi/180*(-1 if item%2 else 1)
            base=update(gt,noise[:,:3],noise[:,3:],gt.new_tensor([d]))
            scene=prepare_scene(e.rgb[0],e.depth[0],base[0],e.mesh,e.k,e.times[0],e.stream,e.cad,f.renderer,fast=True)
            heavy=item%4>=2;oid=f.streams[e.stream.split('|')[0]]['object_id']
            plan=f.occluders.plan(np.random.default_rng(seed+581),oid,heavy=heavy,start=1 if item%4==0 else 0,duration=1,target_fraction=.8 if heavy else .3)
            occ=plan.render(scene,0);obs=encode_scenes(m,[scene],occlusions=[occ])
            with torch.autocast('cuda',dtype=torch.bfloat16):baseline,_=m(obs)
            target=build_teachers(m.ema_teacher,[scene],gt,masks,[occ.mask],f.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
            visible=(crop_images_fast(masks.float(),scene.affine,mode='nearest')>.5)&scene.bounds&~occ.mask
            geometry=obs.geometry_image.float()
            available=(geometry[:,3:4]>0)&scene.bounds&torch.isfinite(geometry[:,2:7]).all(1,keepdim=True)
            reference=torch.cat((geometry[:,4:7],geometry[:,2:3]),1)
            fallback=torch.cat((baseline['surface_xyz'],baseline['surface_depth_residual']),1)
            continuous=baseline['atlas_fallback'][:,:4]
            from lip.unified.flow_reconstruction import template_points
            anchors=template_points(geometry,obs.cad_valid)
            anchor=torch.cat((anchors['xyz'],anchors['depth']),-1).transpose(1,2).reshape(1,4,16,16)
            sampled=anchor.repeat_interleave(14,-2).repeat_interleave(14,-1)
            mass=F.avg_pool2d(available.float(),14)
            mean=F.avg_pool2d(reference*available,14)/mass.clamp_min(1e-8)
            averaged=mean.repeat_interleave(14,-2).repeat_interleave(14,-1)
            variants=dict(network=fallback,continuous=continuous,
                dense_reference=torch.where(available,reference,fallback),
                patch_sample=torch.where(available,sampled,fallback),
                patch_mean=torch.where(available,averaged,fallback))
            row=dict(seed=seed,stream=e.stream,angle=angle,heavy=heavy,variants={})
            for name,prediction in variants.items():
                values={}
                for region,mask in [('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight),('observed',visible)]:
                    canonical=mask&target.cad_geometry_valid
                    if not canonical.any():continue
                    covered=canonical&available;uncovered=canonical&~available
                    xyz=(prediction[:,:3]-target.cad_geometry_xyz).norm(dim=1,keepdim=True)*d*1000
                    depth=(prediction[:,3:4]-target.surface_depth_residual).abs()*d*1000
                    avg=lambda value,selected:float(value[selected].mean()) if selected.any() else None
                    values[region]=dict(pixels=int(canonical.sum()),covered_pixels=int(covered.sum()),
                        canonical_xyz_mm=avg(xyz,canonical),depth_mm=avg(depth,canonical),
                        covered_xyz_mm=avg(xyz,covered),uncovered_xyz_mm=avg(xyz,uncovered),
                        covered_depth_mm=avg(depth,covered),uncovered_depth_mm=avg(depth,uncovered))
                row['variants'][name]=values
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
    (a.out/'receipt.json').write_text(json.dumps(dict(completed=True,records=len(rows),source_sha256=DIGEST,
        model_frozen=True,physical_holdout=True,training_partition_only=True,teacher_input=False,
        scope='Input reference uses estimated pose, including a controlled zero-error arm. Same target masks for all variants; unavailable reference pixels use network fallback. Patch variants use identical dense-reference coverage.',
        seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':main()
