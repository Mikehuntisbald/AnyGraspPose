"""Fixed40 controlled pose response; separate from GT-free native validation."""
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.features import prepare_scene,encode_scenes
from lip.unified.serial_completion import pack_completion
from lip.unified.renderer import FullTextureRenderer
from lip.geometry.so3 import center_pose,update,angle,log
from lip.engine.jepa_checkpoint import load_core,sha
from lip.evaluation.metrics import errors


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(0);cv2.setNumThreads(0)
    c=yaml.safe_load(Path(a.config).read_text());model=build_model(c)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);load_core(model,saved['model']);del saved
    model.requires_grad_(False).eval();model.fast_geometry=model.vector_geometry=True
    store=make_store(c,model);renderer=FullTextureRenderer('cuda')
    root=Path(c['paths']['data_root']);index=Path(c['paths']['index_root']);audit=json.loads((index/'audit.json').read_text())
    registry={x['stream_id']:x for x in map(json.loads,(index/'streams.jsonl').read_text().splitlines()) if x['split']=='val'}
    init=json.loads(Path(c['paths']['val_initializers']).read_text())['initializers']
    old=Path(c['paths']['baseline_scored'])/'predictions.jsonl'
    reference={(x['stream_id'],x['frame_index']):x for x in map(json.loads,old.read_text().splitlines())}
    info=json.loads(Path(c['cad_surface']['models_info']).read_text());chosen={}
    for sid,s in sorted(registry.items()):
        if init[sid] is not None and s['num_frames']-init[sid]['frame_index']>=40:chosen.setdefault('/'.join(sid.split('/')[:2]),sid)
    assert len(chosen)==40
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rows=0;begun=time.monotonic()
    with torch.no_grad(),(out/'frames.jsonl').open('w') as writer:
        for physical,sid in sorted(chosen.items())[a.rank::a.world]:
            s=registry[sid];first=init[sid]['frame_index'];directory=root/s['relative_dir']
            with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            center=torch.tensor(mesh['center'],device='cuda');d=float(mesh['diameter']);k=torch.tensor(s['intrinsics'],device='cuda')
            cad=store.get(root/s['mesh_path'],mesh)
            with np.load(index/s['pose_cache']) as z:truth=center_pose(torch.tensor(z['poses'],device='cuda'),center)
            pool=list(range(first+8,s['num_frames']));worst=min(pool,key=lambda f:2. if reference[sid,f]['visibility'] is None else reference[sid,f]['visibility'])
            frames=sorted(set((first+8,min(first+24,s['num_frames']-1),worst)))
            symmetry=bool(info[str(s['object_id'])].get('symmetries_discrete') or info[str(s['object_id'])].get('symmetries_continuous'))
            for frame in frames:
                im=cv2.imread(str(directory/f'color_{frame:06d}.jpg'));dep=cv2.imread(str(directory/f'aligned_depth_to_color_{frame:06d}.png'),-1)
                rgb=torch.tensor(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy(),device='cuda').float()/255
                depth=torch.tensor(dep.astype('f4')[None],device='cuda')*audit['depth_scale_to_m'];gt=truth[frame]
                cases=[('zero',gt)]
                for axis in range(6):
                    for sign in (-1,1):
                        delta=gt.new_zeros(1,6);delta[0,axis]=sign*(torch.pi/18 if axis<3 else .05)
                        cases.append((f'axis{axis}_{sign:+d}',update(gt[None],delta[:,:3],delta[:,3:],gt.new_tensor([d]))[0]))
                for name,base in cases:
                    scene=prepare_scene(rgb,depth,base,mesh,k,frame/audit['fps'],sid,cad,renderer,fast=True)
                    obs=encode_scenes(model,[scene]);valid=F.avg_pool2d(obs.packet.pixel_valid.float(),14,14).flatten(1)>=.999
                    with torch.autocast('cuda',dtype=torch.bfloat16):result,_=model(obs)
                    packet=pack_completion(result['f_mid_predicted'],result['f_predicted'],torch.cat((result['surface_xyz'],result['surface_depth_residual'],result['geometry_valid_logits']),1),
                        obs.geometry_image,obs.crop_rays,obs.base,obs.diameter,result['evidence_logits'],result['support_logits'],obs.mid,obs.last,valid,obs.measured_depth_m)
                    def read(pack):
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            obj,metrics=model.read_completion(pack,obs.base)
                            delta=model.head(F.layer_norm(obj[:,0],(256,))).float()*metrics['serial_residual_scale'][:,None]*metrics['pose_evidence_available'][:,None]
                        return update(obs.base,delta[:,:3],delta[:,3:],obs.diameter)[0]
                    poses={'base':base,'predicted':result['pose_centered'][0]}
                    if name in ('zero','axis0_+1','axis1_+1','axis2_+1'):
                        rendered=renderer(cad['appearance'],gt,scene.k_crop,224);mask=rendered['mask'][None,None]&scene.bounds
                        oracle=dict(packet,xyz=rendered['xyz'][None]/d,
                            camera=obs.crop_rays*rendered['depth'][None]/d-base[None,:3,3,None,None]/d,
                            weight=mask.float(),measured_weight=torch.zeros_like(mask,dtype=torch.float32),completed_weight=mask.float())
                        poses['oracle_geometry']=read(oracle)
                        off=dict(packet,weight=packet['measured_weight'],completed_weight=torch.zeros_like(packet['weight']))
                        poses['completion_off']=read(off)
                        feature_off=dict(packet,feature=torch.zeros_like(packet['feature']))
                        poses['appearance_off']=read(feature_off)
                    expected=torch.cat((log(gt[:3,:3]@base[:3,:3].T),(gt[:3,3]-base[:3,3])/d))
                    pred=torch.cat((result['delta_rotvec'][0],result['delta_center_norm'][0])).float()
                    row=dict(physical=physical,stream_id=sid,frame=frame,object_id=s['object_id'],symmetric=symmetry,visibility=reference[sid,frame]['visibility'],condition=name,
                        expected_delta=expected.cpu().tolist(),predicted_delta=pred.cpu().tolist(),
                        metrics={key:errors(pose.cpu().numpy(),gt.cpu().numpy(),mesh['points'],d,dtype='f4') for key,pose in poses.items()})
                    writer.write(json.dumps(row,allow_nan=False)+'\n');rows+=1
            writer.flush();print(json.dumps(dict(rank=a.rank,physical=physical,rows=rows)),flush=True)
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,rows=rows,checkpoint_sha256=sha(a.checkpoint),rank=a.rank,world=a.world,
        seconds=time.monotonic()-begun,oracle_conditions=True,official_test_access=False,optimizer_updates=0),indent=2)+'\n')

if __name__=='__main__':main()
