"""Train-only routing diagnostic on three fixed frames; labels used only for scoring."""
from pathlib import Path
import sys,json,torch,yaml,argparse
sys.path.insert(0,str(Path.cwd()/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.reconstruction_only import initialize_training
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
c=yaml.safe_load(args.config.read_text());c['runtime']['compile_frame']=False
model,source,reference=None,None,None
model=build_model(c);source,receipt,reference=initialize_training(model,c,8);del source
factory=Factory(c,model,make_store(c,model));inputs,targets=zip(*(factory.sample(42000139+i,frames=40) for i in range(4)))
model.eval()
with torch.no_grad():
 for frame in (0,8,16):
  scenes=[prepare_scene(e.rgb[frame],e.depth[frame],e.initial,e.mesh,e.k,e.times[frame],e.stream,e.cad,factory.renderer) for e in inputs]
  occlusions=[e.occlusion_plan.render(s,frame) for e,s in zip(inputs,scenes)]
  obs=encode_scenes(model,scenes,frame_id=frame,occlusions=occlusions)
  with torch.autocast('cuda',dtype=torch.bfloat16):out,_=model(obs)
  vis=out['evidence_logits'].sigmoid();prob=torch.nn.functional.avg_pool2d(out['coarse_surface'][:,4:5].sigmoid(),14,14).flatten(1)
  quality=obs.rope_depth_stats
  t=build_teachers(model.ema_teacher,scenes,torch.stack([x[0][frame] for x in targets]),torch.stack([x[1][frame] for x in targets]),[o.mask for o in occlusions],factory.renderer,real_geometry_max_radius_d=1.)
  result=dict(frame=frame,visibility_max=float(vis.max()),recovered_valid_max=float(prob.max()),visible_counts={str(v):int((vis>=v).sum()) for v in (.3,.5,.6,.7)},recovered_counts={str(v):int((prob>=v).sum()) for v in (.3,.5,.6,.7)},quality_mass_ok=int((quality[...,0]>=.5).sum()),quality_std_ok=int((quality[...,1]<=.05).sum()),measured=int(out['rope_measured_mask'].sum()),recovered=int(out['rope_recovered_mask'].sum()))
  result['pixel_validity_max']=float(out['coarse_surface'][:,4:5].sigmoid().max())
  result['pixel_validity_gt_pos_mean']=float(out['coarse_surface'][:,4:5].sigmoid()[t.geometry_valid_label==1].mean())
  result['pixel_validity_gt_neg_mean']=float(out['coarse_surface'][:,4:5].sigmoid()[t.geometry_valid_label==0].mean())
  result['support_max']=float(out['support_logits'].sigmoid().max())
  result['visible_precision']={str(v):float(t.visible_label.nan_to_num()[vis>=v].mean()) if (vis>=v).any() else None for v in (.3,.5,.6,.7)}
  print(json.dumps(result),flush=True)
