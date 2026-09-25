"""Training-split frozen JEPA cache for a bounded dense CAD-readout comparison."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.unified.execution_speed import crop_images_fast
from lip.engine.jepa_checkpoint import load_core,sha
from lip.geometry.so3 import update

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8);p.add_argument('--records',type=int,default=32);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(42)
    c=yaml.safe_load(Path(a.config).read_text());m=build_model(c)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);load_core(m,saved['model']);del saved
    m.requires_grad_(False).eval();m.fast_geometry=m.vector_geometry=True;factory=Factory(c,m,make_store(c,m))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=[];start=time.monotonic();captured=[]
    hook=m.surface_head.register_forward_pre_hook(lambda module,args:captured.append(([x.detach() for x in args[0]],args[1].detach())))
    with torch.no_grad():
        for item in range(a.records):
            seed=19000000+a.rank+item*a.world;e,(truth,visible)=factory.sample(seed,frames=12);frame=8
            oid=factory.streams[e.stream.split('|')[0]]['object_id']
            e.occlusion_plan=factory.occluders.plan(np.random.default_rng(seed),oid,heavy=True,start=4,duration=8,target_fraction=.8)
            d=float(e.mesh['diameter']);gt=truth[frame:frame+1]
            noise=gt.new_zeros(1,6);noise[0,item%3]=(-1 if item%2 else 1)*torch.pi/18
            base=update(gt,noise[:,:3],noise[:,3:],gt.new_tensor([d]))[0]
            scene=prepare_scene(e.rgb[frame],e.depth[frame],base,e.mesh,e.k,e.times[frame],e.stream,e.cad,factory.renderer,fast=True)
            occ=e.occlusion_plan.render(scene,frame);obs=encode_scenes(m,[scene],occlusions=[occ])
            target=build_teachers(m.ema_teacher,[scene],gt,visible[frame:frame+1],[occ.mask],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True)
            vm=(crop_images_fast(visible[frame:frame+1].float(),scene.affine,mode='nearest')>.5)&~occ.mask&scene.bounds
            captured.clear()
            with torch.autocast('cuda',dtype=torch.bfloat16):output,_=m(obs)
            levels,valid=captured[-1]
            measured_xyz=torch.einsum('bji,bjhw->bihw',gt[:,:3,:3],target.camera_rays*obs.measured_depth_m/obs.diameter[:,None,None,None]-target.camera_translation_d[:,:,None,None])
            observed_valid=vm&(obs.measured_depth_m>0)&torch.isfinite(obs.measured_depth_m)&(measured_xyz.square().sum(1,keepdim=True)<1.)
            cpu=lambda x:x.detach().cpu()
            cache.append(dict(seed=seed,stream=e.stream,levels=[cpu(x) for x in levels],valid=cpu(valid),diameter=d,
                xyz=cpu(target.surface_xyz),depth=cpu(target.surface_depth_residual),validity=cpu(target.geometry_valid_label),
                real_mask=cpu(target.geometry_real_weight),proxy_mask=cpu(target.geometry_proxy_weight),
                observed_xyz=cpu(measured_xyz),observed_mask=cpu(observed_valid),rotation=cpu(gt[:,:3,:3]),
                translation_d=cpu(target.camera_translation_d),rays=cpu(target.camera_rays),base_depth_d=cpu(target.base_depth_d),
                cad_features=cpu(obs.cad_surface_features),cad_geometry=cpu(obs.cad_surface_geometry),cad_available=cpu(obs.cad_surface_valid)))
            del output
            print(json.dumps(dict(rank=a.rank,records=len(cache),seconds=time.monotonic()-start)),flush=True)
    hook.remove();torch.save(cache,out/'decoder_cache.pt')
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,checkpoint_sha256=sha(a.checkpoint),records=len(cache),optimizer_updates=0,
        training_split_only=True,official_test_access=False,source_kind='V21 frozen JEPA four levels and fixed coarse routing',cache_sha256=sha(out/'decoder_cache.pt')),indent=2)+'\n')

if __name__=='__main__':main()
