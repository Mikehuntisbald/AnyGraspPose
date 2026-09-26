"""V63 final DPT input sensitivity. No oracle input and no model updates."""
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
            seed=63050000+a.rank+8*draw;draw+=1;e,(truth,masks)=f.sample(seed,frames=1)
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
            row=dict(seed=seed,stream=e.stream,angle=angle,heavy=heavy,variants={})
            dense=m.surface_head.dense_features
            final_levels=[];counts=[0]
            def flow_count(module,args,output):counts[0]+=1
            count_hook=m.flow_reconstruction.register_forward_hook(flow_count)
            try:
                for variant in ('normal','roll0','roll1','roll2','roll3','gradients'):
                    counts[0]=0;final_levels.clear()
                    def decode(levels,valid):
                        if counts[0]!=2:return dense(levels,valid)
                        if variant=='gradients':
                            levels=[x.detach().clone().requires_grad_() for x in levels]
                        elif variant.startswith('roll'):
                            index=int(variant[-1]);levels=list(levels)
                            x=levels[index];levels[index]=x.reshape(len(x),16,16,256).roll(1,2).reshape_as(x)
                        final_levels.extend(levels)
                        return dense(levels,valid)
                    m.surface_head.dense_features=decode
                    with torch.set_grad_enabled(variant=='gradients'),torch.autocast('cuda',dtype=torch.bfloat16):out,_=m(obs)
                    assert counts[0]==2 and len(final_levels)==4
                    if variant in ('normal','gradients'):
                        assert torch.equal(out['surface_xyz'],baseline['surface_xyz'])
                        assert torch.equal(out['surface_depth_residual'],baseline['surface_depth_residual'])
                    result=dict(regions={})
                    for region,mask in [('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight)]:
                        if not mask.any():continue
                        canonical=mask&target.cad_geometry_valid
                        result['regions'][region]=dict(pixels=int(mask.sum()),
                            canonical_xyz_mm=float((out['surface_xyz']-target.cad_geometry_xyz).norm(dim=1,keepdim=True)[canonical].mean()*d*1000) if canonical.any() else None,
                            depth_mm=float((out['surface_depth_residual']-target.surface_depth_residual).abs()[mask].mean()*d*1000),
                            xyz_change_mm=float((out['surface_xyz']-baseline['surface_xyz']).norm(dim=1,keepdim=True)[mask].mean()*d*1000),
                            depth_change_mm=float((out['surface_depth_residual']-baseline['surface_depth_residual']).abs()[mask].mean()*d*1000))
                    if variant=='gradients':
                        mask=(target.geometry_real_weight|target.geometry_proxy_weight|visible)&target.cad_geometry_valid
                        with torch.enable_grad():
                            depth=(out['surface_depth_residual']-target.surface_depth_residual).abs()[mask].mean()
                            # XYZ is a straight-through CAD-point readout: this gradient
                            # is its local surrogate, not the hard-index derivative.
                            xyz=(out['surface_xyz']-target.cad_geometry_xyz).square().sum(1,keepdim=True)[mask].mean()
                            result['sensitivity']={}
                            for name,loss in [('depth',depth),('xyz_surrogate',xyz)]:
                                grads=torch.autograd.grad(loss,final_levels,retain_graph=True)
                                score=torch.stack([g.float().norm()*x.detach().float().norm() for g,x in zip(grads,final_levels)])
                                assert torch.isfinite(score).all()
                                result['sensitivity'][name]=dict(scaled_norm=score.cpu().tolist(),fraction=(score/score.sum().clamp_min(1e-20)).cpu().tolist())
                    row['variants'][variant]=result
                    m.surface_head.dense_features=dense
            finally:
                count_hook.remove();m.surface_head.dense_features=dense
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
    (a.out/'receipt.json').write_text(json.dumps(dict(completed=True,records=len(rows),source_sha256=DIGEST,
        model_frozen=True,physical_holdout=True,training_partition_only=True,normal_rewrite_bitwise_verified=True,
        scope='Only final DPT invocation is intervened. Each roll moves one input map by one14px patch horizontally; not a trained replacement. Gradient scaling multiplies input and gradient norms. XYZ gradient is the existing local straight-through surrogate.',
        seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':main()
