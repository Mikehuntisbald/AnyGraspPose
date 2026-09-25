"""Fixed40 controlled pose response; separate from GT-free native validation."""
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.features import prepare_scene,encode_scenes
from lip.unified.serial_completion import pack_completion
from lip.unified.execution_speed import crop_images_fast
from lip.unified.renderer import FullTextureRenderer
from lip.geometry.so3 import center_pose,update,angle,log
from lip.engine.jepa_checkpoint import load_core,sha
from lip.evaluation.metrics import errors


def diagnostic_rigid_fit(packet,base,diameter,robust=False):
    """Read-only geometric solvability probe; never used in deployed tracker."""
    x=packet['xyz'].double().flatten(2).transpose(1,2)
    y=packet['camera'].double().flatten(2).transpose(1,2)
    initial=packet['weight'].double().flatten(1)
    w=initial
    for iteration in range(4 if robust else 1):
        wn=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
        mx=(x*wn[...,None]).sum(1);my=(y*wn[...,None]).sum(1)
        cov=(x-mx[:,None]).transpose(1,2)@((y-my[:,None])*wn[...,None])
        u,sv,vh=torch.linalg.svd(cov)
        fix=torch.eye(3,device=x.device,dtype=x.dtype)[None].repeat(len(x),1,1)
        fix[:,2,2]=torch.linalg.det(vh.transpose(-1,-2)@u.transpose(-1,-2))
        rotation=vh.transpose(-1,-2)@fix@u.transpose(-1,-2)
        translation=my-(rotation@mx[...,None]).squeeze(-1)
        residual=(x@rotation.transpose(-1,-2)+translation[:,None]-y).norm(dim=-1)
        # Fixed robust scale 1 cm; not tuned using GT residuals.
        w=initial/(1+(residual*diameter[:,None]/.01).square())
    pose=base.double().clone();pose[:,:3,:3]=rotation
    pose[:,:3,3]=base[:,:3,3]+diameter[:,None]*translation
    return pose[0].float()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8);p.add_argument('--geometry-components',action='store_true');p.add_argument('--readout-checkpoint');a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(0);cv2.setNumThreads(0)
    c=yaml.safe_load(Path(a.config).read_text());model=build_model(c)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);load_core(model,saved['model']);del saved
    readout_identity=None
    if a.readout_checkpoint:
        from lip.unified.serial_completion import CompletionRelations
        from lip.unified.reconstruction_only import is_pose_parameter
        saved=torch.load(a.readout_checkpoint,map_location='cpu',weights_only=False)
        if saved['source_sha256']!=sha(a.checkpoint):raise ValueError('Readout source mismatch')
        model.geometry_readout=CompletionRelations(saved['conditioning']).cuda()
        states=model.state_dict()
        expected={k for k in states if is_pose_parameter(k)}
        if set(saved['model'])!=expected:raise ValueError('Readout parameter keys mismatch')
        for k,v in saved['model'].items():states[k].copy_(v)
        readout_identity=dict(sha256=sha(a.readout_checkpoint),conditioning=saved['conditioning'],step=saved['step'])
        del saved,states
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
                if a.geometry_components:cases=[x for x in cases if x[0] in ('zero','axis0_+1','axis1_+1','axis2_+1')]
                for name,base in cases:
                    scene=prepare_scene(rgb,depth,base,mesh,k,frame/audit['fps'],sid,cad,renderer,fast=True)
                    obs=encode_scenes(model,[scene]);valid=F.avg_pool2d(obs.packet.pixel_valid.float(),14,14).flatten(1)>=.999
                    with torch.autocast('cuda',dtype=torch.bfloat16):result,_=model(obs)
                    packet=pack_completion(result['f_mid_predicted'],result['f_predicted'],torch.cat((result['surface_xyz'],result['surface_depth_residual'],result['geometry_valid_logits']),1),
                        obs.geometry_image,obs.crop_rays,obs.base,obs.diameter,result['evidence_logits'],result['support_logits'],obs.mid,obs.last,valid,obs.measured_depth_m,feature_source=getattr(model,'readout_feature_source','observed_visible'))
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
                        if a.geometry_components:
                            # Keep all original confidence/support and real measurement ownership.
                            # GT interventions diagnose a bottleneck; these are never deployment inputs.
                            eligible=mask & (packet['weight']>0)
                            missing=eligible & (packet['measured_weight']==0)
                            target_xyz=rendered['xyz'][None]/d
                            target_camera=obs.crop_rays*rendered['depth'][None]/d-base[None,:3,3,None,None]/d
                            fixed_xyz=torch.where(eligible,target_xyz,packet['xyz'])
                            fixed_camera=torch.where(missing,target_camera,packet['camera'])
                            poses['gt_xyz_fixed_support']=read(dict(packet,xyz=fixed_xyz))
                            poses['gt_missing_depth_fixed_support']=read(dict(packet,camera=fixed_camera))
                            poses['gt_both_fixed_support']=read(dict(packet,xyz=fixed_xyz,camera=fixed_camera))
                            # Allowed estimated-pose render: fill only overlap with predicted missing area.
                            render_mask=scene.render['mask'][None,None]&scene.bounds
                            reference_camera=obs.crop_rays*scene.render['depth'][None]/d-base[None,:3,3,None,None]/d
                            cad_missing=render_mask & (packet['measured_weight']==0)
                            pruned={k:(v*mask if k in ('weight','measured_weight','completed_weight') else v) for k,v in packet.items()}
                            poses['gt_support_only']=read(pruned)
                            poses['gt_both_pruned_support']=read(dict(pruned,xyz=fixed_xyz,camera=fixed_camera))
                            with np.load(directory/f'labels_{frame:06d}.npz') as labels:
                                visible=torch.tensor((labels['seg']==s['object_id']).copy(),device='cuda')[None,None].float()
                            visible=crop_images_fast(visible,scene.affine,mode='nearest')>.5
                            true_measured=visible&(obs.measured_depth_m>0)&scene.bounds
                            recovered_camera=obs.crop_rays*result['surface_depth_m']/d-base[None,:3,3,None,None]/d
                            reliable_camera=torch.where(true_measured,packet['camera'],recovered_camera)
                            clean=dict(pruned,camera=reliable_camera,
                                measured_weight=pruned['measured_weight']*true_measured,
                                completed_weight=pruned['weight']-pruned['measured_weight']*true_measured)
                            poses['gt_support_reject_false_measured']=read(clean)
                            ideal_camera=torch.where(true_measured,obs.crop_rays*obs.measured_depth_m/d-base[None,:3,3,None,None]/d,target_camera)
                            poses['gt_both_pruned_true_measured']=read(dict(clean,xyz=fixed_xyz,camera=ideal_camera))
                            contamination=dict(outside_support_weight=float((packet['weight']*~mask).sum()/packet['weight'].sum().clamp_min(1e-6)),
                                false_measured_weight=float((packet['measured_weight']*~visible).sum()/packet['weight'].sum().clamp_min(1e-6)))
                            poses['cad_missing_depth_fixed_support']=read(dict(packet,camera=torch.where(cad_missing,reference_camera,packet['camera'])))
                            poses['gt_xyz_cad_missing_depth_fixed_support']=read(dict(packet,xyz=fixed_xyz,camera=torch.where(cad_missing,reference_camera,packet['camera'])))
                            # Sensor-consistent oracle canonical coordinates on genuinely measured pixels.
                            real_xyz=torch.einsum('ij,bjhw->bihw',gt[:3,:3].T,obs.crop_rays*obs.measured_depth_m/d-gt[None,:3,3,None,None]/d)
                            sensor_xyz=torch.where(true_measured,real_xyz,target_xyz)
                            measured_kept=true_measured&(packet['measured_weight']>0)
                            ideal_sensor=dict(clean,xyz=torch.where(measured_kept,real_xyz,target_xyz),camera=torch.where(measured_kept,packet['camera'],target_camera))
                            poses['gt_sensor_consistent_fixed_weight']=read(ideal_sensor)
                            # Diagnostic error-dependent confidence, not a deployed confidence predictor.
                            xyz_error=(packet['xyz']-sensor_xyz).norm(dim=1,keepdim=True)*d
                            camera_error=(packet['camera']-torch.where(measured_kept,packet['camera'],target_camera)).norm(dim=1,keepdim=True)*d
                            quality=mask.float()/(1+(xyz_error/.01).square()+(camera_error/.01).square())
                            quality_packet={k:(v*quality if k in ('weight','measured_weight','completed_weight') else v) for k,v in packet.items()}
                            poses['gt_error_quality_only']=read(quality_packet)
                            contamination['quality_retained_weight']=float(quality_packet['weight'].sum()/packet['weight'].sum().clamp_min(1e-6))
                            for fit_name,fit_packet in [('sensor_consistent_oracle',ideal_sensor),('gt_error_quality_only',quality_packet),('predicted',packet),('oracle_geometry',oracle),('gt_both_pruned_support',dict(pruned,xyz=fixed_xyz,camera=fixed_camera))]:
                                poses['rigid_'+fit_name]=diagnostic_rigid_fit(fit_packet,obs.base,obs.diameter)
                                poses['robust_rigid_'+fit_name]=diagnostic_rigid_fit(fit_packet,obs.base,obs.diameter,True)
                    expected=torch.cat((log(gt[:3,:3]@base[:3,:3].T),(gt[:3,3]-base[:3,3])/d))
                    pred=torch.cat((result['delta_rotvec'][0],result['delta_center_norm'][0])).float()
                    row=dict(physical=physical,stream_id=sid,frame=frame,object_id=s['object_id'],symmetric=symmetry,visibility=reference[sid,frame]['visibility'],condition=name,
                        expected_delta=expected.cpu().tolist(),predicted_delta=pred.cpu().tolist(),
                        metrics={key:errors(pose.cpu().numpy(),gt.cpu().numpy(),mesh['points'],d,dtype='f4') for key,pose in poses.items()})
                    if a.geometry_components:row['contamination']=contamination
                    writer.write(json.dumps(row,allow_nan=False)+'\n');rows+=1
            writer.flush();print(json.dumps(dict(rank=a.rank,physical=physical,rows=rows)),flush=True)
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,rows=rows,checkpoint_sha256=sha(a.checkpoint),rank=a.rank,world=a.world,
        readout_override=readout_identity,geometry_components=a.geometry_components,seconds=time.monotonic()-begun,oracle_conditions=True,official_test_access=False,optimizer_updates=0),indent=2)+'\n')

if __name__=='__main__':main()
