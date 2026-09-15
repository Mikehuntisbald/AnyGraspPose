"""Single-image s0-test refinements of released PoseCNN predictions; no GT pose/mask reads."""
import argparse,csv,json,logging,os,sys,time
from pathlib import Path
import cv2
import numpy as np
import torch
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.benchmark.bop_io import load_predictions,select_initializers,csv_row,FIELDS,key
from lip.engine.stream_config import make_model,load_stream_config
from lip.engine.stream_checkpoint import sha,source_hash,load_init
from lip.engine.config import check_data_gate
from lip.engine.stream_runtime import valid_pose
from lip.geometry.mesh import mesh_metadata
from lip.geometry.renderer import Renderer
from lip.data.index import CLASSES
from lip.integrations.foundationpose import FoundationPoseAdapter


def main():
 p=argparse.ArgumentParser(__doc__)
 for name in ['protocol','out']:p.add_argument('--'+name,required=True,type=Path)
 p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8);a=p.parse_args()
 protocol=json.loads(a.protocol.read_text());assert protocol['frozen'] and protocol['split']=='s0_test' and protocol['history']=='none'
 assert protocol['source_sha256']==source_hash() and protocol['inference_sha256']==sha(__file__)
 raw_root=Path(protocol['data_root']).resolve()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  path=Path(os.fsdecode(args[0])).resolve()
  if not path.is_relative_to(raw_root):return
  if path.name.startswith(('labels_','scene_gt')) or any(x in ('mask','mask_visib') or x.startswith('mano') for x in path.parts):
   raise PermissionError('Inference attempted annotation access: '+str(path))
  if args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):raise PermissionError('Raw data is read-only')
 sys.addaudithook(guard)
 assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256'] and sha(protocol['initializer_csv'])==protocol['initializer_sha256']
 assert sha(protocol['targets'])==protocol['targets_sha256']
 assert 0<=a.rank<a.world
 a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);cv2.setNumThreads(0);torch.manual_seed(42);np.random.seed(42)
 c=load_stream_config(protocol['config']);model=make_model(c).cuda().eval();load_init(protocol['checkpoint'],model,check_data_gate(protocol['index_root']),c)
 assert c['architecture_id']=='stream_dual_cross_residual'
 targets=json.loads(Path(protocol['targets']).read_text());targets=[r for r in targets if r['scene_id']%a.world==a.rank]
 selected,missing=select_initializers(load_predictions(protocol['initializer_csv']),targets)
 # Group by object to amortize native mesh setup; never carry image/pose state.
 selected.sort(key=lambda r:(r['obj_id'],r['scene_id'],r['im_id']))
 fp_root=Path(protocol['fp_root']);sys.path.insert(0,str(fp_root))
 from estimater import FoundationPose
 from learning.training.predict_pose_refine import PoseRefinePredictor
 import nvdiffrast.torch as dr
 class RefineOnly(FoundationPose):
  def make_rotation_grid(self,*args,**kwargs):pass
 logging.getLogger().setLevel(logging.WARNING)
 refiner=PoseRefinePredictor();glctx=dr.RasterizeCudaContext();renderer=Renderer('cuda');est=None;oid=None;camera={}
 methods=['posecnn','posecnn_lip','posecnn_fp','posecnn_lip_fp'];files={m:(a.out/(m+'.csv')).open('w') for m in methods}
 writers={m:csv.DictWriter(f,fieldnames=FIELDS) for m,f in files.items()}
 for w in writers.values():w.writeheader()
 receipt=dict(completed=False,rank=a.rank,world=a.world,protocol_sha256=sha(a.protocol),source_sha256=source_hash(),targets=len(targets),estimates=len(selected),missing=missing,gt_pose_reads=0,gt_mask_reads=0,hand_annotation_reads=0,history='none',failures={})
 (a.out/'manifest.json').write_text(json.dumps(receipt,indent=2))
 begin=time.time()
 with torch.no_grad(),(a.out/'events.jsonl').open('w') as events:
  for i,row in enumerate(selected):
   if oid!=row['obj_id']:
    oid=row['obj_id'];path=Path(protocol['data_root'])/'models'/CLASSES[oid-1]/'textured_simple.obj'
    mesh,_=mesh_metadata(path,Path(protocol['mesh_cache'])/str(a.rank),scale=1.)
    raw=trimesh.load(path,process=False,force='mesh')
    if est is None:est=RefineOnly(raw.vertices,raw.vertex_normals,mesh=raw,scorer=object(),refiner=refiner,glctx=glctx,debug=0,debug_dir=str(a.out/'fp_debug'))
    else:est.reset_object(raw.vertices,raw.vertex_normals,mesh=raw)
    est.diameter=float(est.diameter);adapter=FoundationPoseAdapter(est)
   scene=Path(protocol['data_root'])/'bop/s0/test'/f"{row['scene_id']:06d}"
   if row['scene_id'] not in camera:camera[row['scene_id']]=json.loads((scene/'scene_camera.json').read_text())
   info=camera[row['scene_id']][str(row['im_id'])];K=np.asarray(info['cam_K'],dtype=np.float32).reshape(3,3)
   color=cv2.imread(str(scene/'rgb'/f"{row['im_id']:06d}.jpg"));dep=cv2.imread(str(scene/'depth'/f"{row['im_id']:06d}.png"),-1)
   if color is None or dep is None:raise FileNotFoundError('Missing official RGB-D input')
   rgb=cv2.cvtColor(color,cv2.COLOR_BGR2RGB);depth=dep.astype('f4')*float(info['depth_scale'])/1000.
   prior=row['pose'];outputs={'posecnn':prior};status={}
   if valid_pose(torch.from_numpy(prior)):
    try:
     state=model.initialize(prior,mesh,K,f"independent:{row['scene_id']}:{row['im_id']}:{oid}",0.,image_shape=depth.shape)
     assert not state.cache.metadata
     proposal,_=model.step(torch.from_numpy(rgb.transpose(2,0,1).copy()),torch.from_numpy(depth[None]),protocol['synthetic_dt'],state,renderer=renderer,precision=c['precision'],image_size=c['image_size'],crop_expansion=c['crop_expansion'])
     outputs['posecnn_lip']=proposal['pose_original'].cpu().numpy() if proposal['status']=='ok' else prior
     status['lip']=proposal['status']
    except ValueError as error:outputs['posecnn_lip']=prior;status['lip']='invalid_initialization:'+str(error)
    for name,base in [('posecnn_fp',prior),('posecnn_lip_fp',outputs['posecnn_lip'])]:
     result=np.asarray(adapter.refine(base,rgb,depth,K,iteration=protocol['fp_iterations']),dtype=np.float32)
     valid=valid_pose(torch.from_numpy(result));outputs[name]=result if valid else base;status[name]='ok' if valid else 'invalid_refinement_retained_input'
   else:
    outputs.update({m:prior for m in methods if m!='posecnn'});status['initialization']='invalid_pose_retained_for_all_methods'
   for m in methods:writers[m].writerow(csv_row(row,outputs[m]))
   events.write(json.dumps(dict(key=key(row),status=status))+'\n')
   for name,value in status.items():
    if value!='ok':receipt['failures'][name+':'+value]=receipt['failures'].get(name+':'+value,0)+1
   if (i+1)%100==0:
    for f in files.values():f.flush()
    events.flush();print(json.dumps(dict(rank=a.rank,completed=i+1,total=len(selected),seconds=time.time()-begin)),flush=True)
 for f in files.values():f.close()
 receipt.update(completed=True,seconds=time.time()-begin,csv_sha256={m:sha(a.out/(m+'.csv')) for m in methods})
 (a.out/'manifest.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
