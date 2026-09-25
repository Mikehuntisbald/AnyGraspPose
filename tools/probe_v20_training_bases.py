"""Read-only train-input base trajectory replay versus frozen student feedback on32 episodes."""
import argparse,json,sys
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.reconstruction_only import initialize_training
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes
from lip.geometry.so3 import angle
from lip.engine.jepa_checkpoint import load_core,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
 torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(42)
 c=yaml.safe_load(Path(a.config).read_text());c['runtime']['compile_frame']=False
 m=build_model(c);_,_,ref=initialize_training(m,c,8)
 checkpoint='/mnt/why/dexycb_lip/unified_jepa_20260921/pose_geometry_v20/runs/seed42/last.pt'
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);load_core(m,saved['model']);del saved
 m.requires_grad_(False).eval();ref.requires_grad_(False).eval();m.weights_version=sha(checkpoint)
 for model in (m,ref):model.fast_geometry=True;model.vector_geometry=True;model.trusted_training_inputs=True
 factory=Factory(c,m,make_store(c,m));seeds=[42*1000033+(40400*8+a.rank)*4+i for i in range(4)]
 episodes,targets=zip(*(factory.sample(seed,frames=40) for seed in seeds));gt=torch.stack([t[0] for t in targets]);diam=gt.new_tensor([float(e.mesh['diameter']) for e in episodes])
 bases={n:torch.stack([e.initial for e in episodes]) for n in ('fixed_reference','student_feedback')};previous={n:None for n in bases};memories={n:None for n in bases}
 history=torch.tensor([e.history for e in episodes],device='cuda');enabled=torch.tensor([e.cad_enabled for e in episodes],device='cuda');meta=json.loads(Path(c['cad_surface']['models_info']).read_text());rows=[]
 with torch.no_grad():
  for frame in range(40):
   for name,model in [('fixed_reference',ref),('student_feedback',m)]:
    base=bases[name]
    scenes=[prepare_scene(e.rgb[frame],e.depth[frame],base[i],e.mesh,e.k,e.times[frame],e.stream,e.cad,factory.renderer,None if previous[name] is None else previous[name][i],None if frame==0 else e.times[frame-1],fast=True) for i,e in enumerate(episodes)]
    occlusions=[e.occlusion_plan.render(s,frame) for e,s in zip(episodes,scenes)]
    obs=encode_scenes(model,scenes,cad_enabled=enabled,frame_id=frame,occlusions=occlusions)
    with torch.autocast('cuda',dtype=torch.bfloat16):output,memories[name]=model(obs,memories[name],history)
    if frame:
     for i,e in enumerate(episodes):
      # Episode stream may carry a seed suffix; use mesh CAD identity instead.
      asset=next(p for p in e.cad['_asset_files'] if p.suffix=='.obj');object_name=asset.parent.name
      rows.append(dict(seed=seeds[i],stream=e.stream,object_name=object_name,frame=frame,arm=name,cad_enabled=bool(enabled[i]),
       base_rotation_deg=float(angle(base[i,:3,:3]@gt[i,frame,:3,:3].T))*180/torch.pi,
       base_center_d=float((base[i,:3,3]-gt[i,frame,:3,3]).norm()/diam[i]),
       after_rotation_deg=float(angle(output['pose_centered'][i,:3,:3]@gt[i,frame,:3,:3].T))*180/torch.pi,
       synthetic_occlusion_fraction=float(occlusions[i].mask.float().mean())))
     previous[name]=base;bases[name]=output['pose_centered'].detach()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);(out/'rows.json').write_text(json.dumps(rows))
 (out/'receipt.json').write_text(json.dumps(dict(completed=True,optimizer_updates=0,episodes=4,rows=len(rows),seeds=seeds,source_sha256=sha(checkpoint),scope='Actual training sampler at global40400; original synthetic occlusion policy. Student feedback may change crop-space augmentation realization. This is a training-distribution diagnostic, not native validation.'),indent=2))
if __name__=='__main__':main()
