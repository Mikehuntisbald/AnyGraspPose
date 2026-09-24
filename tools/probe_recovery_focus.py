"""Fixed 40-sequence spatial/geometry recovery probe with three history arms. Labels are confined to teacher/scoring.

Shared crops use the previous frame of the sealed, uncorrupted baseline; pose
scores here are conditional one-step refinements, not closed-loop tracking.
Full native tracking is measured separately by infer_unified_jepa_val.py.
"""
import argparse,json,sys,time,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch,yaml,cv2
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.engine.jepa_checkpoint import load_core,sha
from lip.engine.object_jepa_checkpoint import atomic_json
from lip.jepa.losses import feature_error,normalize
from lip.geometry.so3 import center_pose
from lip.geometry.crop import crop_images
from lip.unified.renderer import FullTextureRenderer as AppearanceRenderer
from lip.unified.occlusion import OccluderBank
from lip.unified.recovery_metrics import feature_diagnostics, geometry_diagnostics, scramble_history


def score(output,obs,teacher,weight,proxy=False,layer_weights=(.25,1.)):
    count=float(weight.sum())
    if count==0:return None
    mid,last=(teacher.proxy_mid,teacher.proxy_last) if proxy else (teacher.real_mid,teacher.real_last)
    candidates=(teacher.proxy_weight>0) if proxy else ((teacher.visible_weight+teacher.hidden_real_weight)>0)
    wm,wl=layer_weights
    metrics={}
    for tag,pred,pmid in [('predicted',output['f_predicted'],output['f_mid_predicted']),('raw',obs.last,obs.mid)]:
        loss=wl*feature_error(pred,last)+wm*feature_error(pmid,mid)
        cosine=F.cosine_similarity(normalize(pred),normalize(last),dim=-1)
        similarity=F.normalize(pred.float(),dim=-1)@F.normalize(last.float(),dim=-1).transpose(-1,-2)
        retrieval=similarity.masked_fill(~candidates[:,None],-1e4).argmax(-1)==torch.arange(256,device=weight.device)[None]
        for name,value in [('feature_loss',loss),('cosine',cosine),('retrieval',retrieval)]:metrics[tag+'_'+name]=float((value*weight).sum()/count)
    return dict(patches=count,**metrics)


def geometry_score(output,teacher,weight,diameter):
    count=float(weight.sum())
    if not count:return None
    xyz=(output['surface_xyz']-teacher.surface_xyz).norm(dim=1,keepdim=True)*diameter*1000
    depth=(output['surface_depth_m']-teacher.surface_depth_m).abs()*1000
    valid=output['geometry_valid_logits'].sigmoid()>=.5
    return dict(pixels=count,xyz_mm=float((xyz*weight).sum()/count),depth_mm=float((depth*weight).sum()/count),
        surface_valid_recall=float((valid*weight).sum()/count))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,default=8)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.cuda.set_device(0 if torch.cuda.device_count()==1 else a.rank);torch.set_num_threads(2);torch.manual_seed(42);cv2.setNumThreads(0)
    c=yaml.safe_load(a.config.read_text());m=build_model(c);record=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    load_core(m,record['model']);m.requires_grad_(False).eval();m.weights_version=sha(a.checkpoint)
    teacher_encoder=getattr(m,'ema_teacher',m.encoder)
    fixed_teacher=c.get('validation',{}).get('fixed_feature_teacher')
    if fixed_teacher:
        import copy
        assert sha(fixed_teacher['checkpoint'])==fixed_teacher['sha256']
        saved=torch.load(fixed_teacher['checkpoint'],map_location='cpu',weights_only=False)
        teacher_encoder=copy.deepcopy(teacher_encoder)
        teacher_encoder.load_state_dict({k.removeprefix('ema_teacher.'):v for k,v in saved['model'].items() if k.startswith('ema_teacher.')},strict=True)
        teacher_encoder.requires_grad_(False).eval();del saved
    layer_weights=(c['training']['loss_weights'].get('feature_mid_weight',.25),c['training']['loss_weights'].get('feature_last_weight',1.))
    store=make_store(c,m);renderer=AppearanceRenderer('cuda')
    index=Path(c['paths']['index_root']);root=Path(c['paths']['data_root']);audit=json.loads((index/'audit.json').read_text())
    bank=OccluderBank(c['paths']['occluder_bank'],audit['split_hash'])
    assert bank.receipt['bank_sha256']==c['augmentation']['bank_sha256']
    streams={s['stream_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines()) if s['split']=='val'}
    initializers=json.loads(Path(c['paths']['val_initializers']).read_text())['initializers']
    reference=Path(c['paths']['baseline_scored'])/'predictions.jsonl'
    poses={(r['stream_id'],r['frame_index']):r for r in map(json.loads,reference.read_text().splitlines())}
    chosen={}
    for sid,s in sorted(streams.items()):
        item=initializers[sid]
        if item is not None and s['num_frames']-item['frame_index']>=40:chosen.setdefault('/'.join(sid.split('/')[:2]),sid)
    assert len(chosen)==40
    selected=sorted(chosen.items())[a.rank::a.world]
    if a.smoke:selected=selected[:1]
    a.out.mkdir(parents=True,exist_ok=False);begun=time.monotonic();rows=0
    manifest=dict(completed=False,checkpoint_sha256=sha(a.checkpoint),reference_sha256=sha(reference),rank=a.rank,world=a.world,
        physical_sequences=[k for k,_ in selected],expected_physical_sequences=sorted(chosen),split='val',smoke=a.smoke,
        teacher_inputs_to_student=False,gt_pose_inputs_to_student=False,official_test_access=False,
        pose_scope='Shared previous-frame baseline estimates; conditional one-step refinement; separate full native validation',
        target_scope='Full GT-CAD image defines proxy features; reconstruction only on naturally-hidden CAD and artificially-hidden real regions; unmasked visible evaluation only',
        history_reconstruction_region=c['validation']['history_reconstruction_region'],
        policies=['on','off','scrambled'], history_intervention='same causal incoming memory; rotate content among valid slots, preserve positions/ages/validity; not unrelated-object history',
        feature_candidates='real: visible plus artificially hidden; proxy: full object interior',occluder_bank_sha256=bank.receipt['bank_sha256'],
        teacher_geometry_max_radius_d=c.get('supervision',{}).get('real_geometry_max_radius_d'),
        severity='Train-only textured hand/other-object RGB-D cutouts; base-silhouette cover light=.25, heavy=.75/.85/.95; actual originally-visible fraction reported',
        temporal_schedule='8 clean observed anchor frames, then 8/16/32 frames of coherent RGB-D foreground occlusion')
    manifest['feature_teacher']='checkpoint EMA DINO' if hasattr(m,'ema_teacher') else 'fixed original DINO'
    manifest['ema_updates']=int(m.ema_updates) if hasattr(m,'ema_updates') else None
    manifest['history_branch_disabled']=getattr(m,'disable_history',False)
    manifest['dino_layers']=dict(student=list(m.encoder.feature_layers),teacher=list(getattr(m,'ema_teacher',m.encoder).feature_layers))
    manifest['feature_layer_weights']=list(layer_weights)
    manifest['fixed_feature_teacher']=fixed_teacher
    manifest['local_structure_diagnostics']=bool(c.get('local_structure'))
    if fixed_teacher:manifest['feature_teacher']='fixed source-checkpoint EMA; independent of checkpoint under evaluation'
    with torch.no_grad(),(a.out/'frames.jsonl').open('w') as f:
        for physical,sid in selected:
            s=streams[sid];first=initializers[sid]['frame_index']
            with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            cad=store.get(root/s['mesh_path'],mesh);k=torch.tensor(s['intrinsics'],device='cuda');center=torch.tensor(mesh['center'],device='cuda')
            with np.load(index/s['pose_cache']) as z:truth=center_pose(torch.tensor(z['poses'][first:first+40],device='cuda'),center)
            seed=int(hashlib.sha256(physical.encode()).hexdigest()[:8],16)+42
            memories={};previous=None
            cases=[('natural',0.,32),('light',.25,8),('heavy8',.75,8),('heavy16',.85,16),('heavy32',.95,32)]
            plans=[bank.plan(np.random.default_rng(seed+i),s['object_id'],heavy=name.startswith('heavy'),
                start=8 if fraction else 1000,duration=duration,target_fraction=fraction) for i,(name,fraction,duration) in enumerate(cases)]
            count=10 if a.smoke else 40
            for t in range(count):
                frame=first+t;directory=root/s['relative_dir'];image=cv2.imread(str(directory/f'color_{frame:06d}.jpg'))
                rgb=torch.tensor(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy(),device='cuda').float()/255
                depth=torch.tensor(cv2.imread(str(directory/f'aligned_depth_to_color_{frame:06d}.png'),-1).astype('f4')[None],device='cuda')*audit['depth_scale_to_m']
                with np.load(directory/f'labels_{frame:06d}.npz') as z:visible=torch.tensor(z['seg']==s['object_id'],device='cuda')[None]
                pose=initializers[sid]['pose_original'] if t==0 else poses[(sid,frame-1)]['pose_original']
                if pose is None:raise ValueError('Missing shared baseline pose')
                base=center_pose(torch.tensor(pose,device='cuda'),center)
                scene=prepare_scene(rgb,depth,base,mesh,k,frame/audit['fps'],sid,cad,renderer,previous,None if t==0 else (frame-1)/audit['fps'])
                previous=base
                teacher=build_teachers(teacher_encoder,[scene],truth[t:t+1],visible[None],[torch.zeros_like(scene.bounds)],renderer,
                    real_geometry_max_radius_d=c.get('supervision',{}).get('real_geometry_max_radius_d'))
                scenes=[replace(scene,stream=sid+'|'+case) for case,_,_ in cases]
                occlusions=[plan.render(case_scene,t) for plan,case_scene in zip(plans,scenes)]
                masks=[o.mask for o in occlusions]
                observations=encode_scenes(m,scenes,frame_id=frame,occlusions=occlusions)
                points=torch.tensor(mesh['points'],device='cuda');gt_points=points@truth[t,:3,:3].T+truth[t,:3,3]
                for lane,((case,fraction,duration),mask) in enumerate(zip(cases,masks)):
                    values={}
                    for name in observations.__dataclass_fields__:
                        value=getattr(observations,name)
                        if name=='packet':
                            values[name]=replace(value,**{n:(tuple([getattr(value,n)[lane]]) if n=='stream_id' else (None if getattr(value,n) is None else getattr(value,n)[lane:lane+1])) for n in value.__dataclass_fields__})
                        elif name=='metadata':values[name]=replace(value,**{n:getattr(value,n)[lane:lane+1] for n in value.__dataclass_fields__})
                        else:values[name]=None if value is None else value[lane:lane+1]
                    obs=type(observations)(**values)
                    added=F.avg_pool2d(mask.float(),14,14).flatten(1)
                    visible_crop=crop_images(visible[None].float(),scene.affine,mode='nearest')>.5
                    retained=float((visible_crop&~mask).sum()/visible_crop.sum().clamp_min(1))
                    target=build_teachers(teacher_encoder,[scene],truth[t:t+1],visible[None],[mask],renderer,
                        encoded_targets=(teacher.real_mid,teacher.real_last,teacher.proxy_mid,teacher.proxy_last),
                        real_geometry_max_radius_d=c.get('supervision',{}).get('real_geometry_max_radius_d'))
                    if hasattr(m,'cad_surface'):
                        from lip.unified.cad_surface_targets import surface_targets,correspondence_score
                        target=surface_targets(m,[scene],target)
                    incoming=memories.get((case,'on'))
                    for policy in ('on','off','scrambled'):
                        read_memory=scramble_history(incoming) if policy=='scrambled' else incoming
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            out,next_memory=m(obs,read_memory,torch.tensor([policy!='off'],device='cuda'))
                        if policy=='on':memories[(case,'on')]=next_memory
                        pred=out['pose_centered'][0];pred_points=points@pred[:3,:3].T+pred[:3,3]
                        adds=torch.cdist(pred_points[None],gt_points[None]).amin(-1).mean()/float(mesh['diameter'])
                        row=dict(physical_sequence=physical,stream_id=sid,frame_index=frame,relative_frame=t,case=case,history=policy,
                            phase='occlusion' if 8<=t<8+duration else 'anchor_or_recovery',target_base_silhouette_fraction=fraction,duration=duration,occluder_kinds=[x[0] for x in occlusions[lane].provenance],
                            remaining_originally_visible_pixel_fraction=retained,pose_adds_d=float(adds),pose_adds_005=100*float(adds<.05),read_memory_tokens=out['read_memory_tokens'],
                            hidden_real=score(out,obs,target,target.hidden_real_weight,layer_weights=layer_weights),visible_real=score(out,obs,target,target.visible_weight,layer_weights=layer_weights),
                            cad_proxy=score(out,obs,target,target.proxy_weight,True,layer_weights=layer_weights),
                            surface_all=geometry_score(out,target,target.geometry_weight,float(mesh['diameter'])),
                            surface_hidden_real=geometry_score(out,target,target.geometry_real_weight,float(mesh['diameter'])),
                            surface_natural_hidden=geometry_score(out,target,target.geometry_hidden_weight,float(mesh['diameter'])),
                            spatial_hidden_real=feature_diagnostics(out,target,target.hidden_real_weight),
                            spatial_visible_real=feature_diagnostics(out,target,target.visible_weight),
                            spatial_cad_proxy=feature_diagnostics(out,target,target.proxy_weight,True),
                            geometry_focus_real=geometry_diagnostics(out,target,target.geometry_real_weight,float(mesh['diameter'])),
                            geometry_focus_proxy=geometry_diagnostics(out,target,target.geometry_proxy_weight,float(mesh['diameter'])))
                        if c.get('dino_layers'):
                            row['spatial_hidden_real_mid']=feature_diagnostics(out,target,target.hidden_real_weight,middle=True)
                            row['spatial_cad_proxy_mid']=feature_diagnostics(out,target,target.proxy_weight,True,middle=True)
                        if hasattr(m,'cad_surface'):
                            row['cad_match_real']=correspondence_score(out,target)
                            row['cad_match_proxy']=correspondence_score(out,target,True)
                        if c.get('local_structure'):
                            from lip.unified.local_structure import local_diagnostics
                            for region,query,candidates,mid,last in (
                                ('real',target.hidden_real_weight,target.visible_weight+target.hidden_real_weight,target.real_mid,target.real_last),
                                ('proxy',target.proxy_weight,target.support_label.nan_to_num()>=.9,target.proxy_mid,target.proxy_last)):
                                for layer,pred,truth_features in (('mid',out['f_mid_predicted'],mid),('last',out['f_predicted'],last)):
                                    row[f'local_{region}_{layer}']=local_diagnostics(pred,truth_features,query,candidates) if query.any() else None
                        f.write(json.dumps(row,allow_nan=False)+'\n');rows+=1
                        if lane in (1,4) and t==8 and policy=='on':
                            # Shared scale for on/off error maps; RGB comes directly from checked targets.
                            real_error=layer_weights[1]*feature_error(out['f_predicted'],teacher.real_last)+layer_weights[0]*feature_error(out['f_mid_predicted'],teacher.real_mid)
                            proxy_error=layer_weights[1]*feature_error(out['f_predicted'],teacher.proxy_last)+layer_weights[0]*feature_error(out['f_mid_predicted'],teacher.proxy_mid)
                            def error_map(value,valid):
                                error=value.reshape(16,16).cpu().numpy();known=valid.reshape(16,16).cpu().numpy()>0
                                heat=cv2.applyColorMap(np.clip(error/2*255,0,255).astype('uint8'),cv2.COLORMAP_INFERNO)
                                heat[~known]=[128,128,128]
                                return cv2.resize(heat,(224,224),interpolation=cv2.INTER_NEAREST)
                            panels=[obs.packet.rgb_crop[0].float().cpu().numpy().transpose(1,2,0),teacher.real_rgb[0].cpu().numpy().transpose(1,2,0),teacher.proxy_rgb[0].cpu().numpy().transpose(1,2,0)]
                            # Undo encoder normalization for the student panel.
                            panels[0]=panels[0]*np.array([.229,.224,.225])+np.array([.485,.456,.406])
                            rgbpanel=np.concatenate([np.clip(v*255,0,255).astype('uint8') for v in panels],axis=1)
                            heat=error_map(real_error,target.hidden_real_weight);proxy_heat=error_map(proxy_error,target.proxy_weight)
                            stem=hashlib.sha256(sid.encode()).hexdigest()[:8]+'_'+case
                            picture=np.concatenate([cv2.cvtColor(rgbpanel,cv2.COLOR_RGB2BGR),heat,proxy_heat],axis=1)
                            header=np.zeros((28,picture.shape[1],3),dtype='uint8')
                            for column,label in enumerate(('Student RGB','Original RGB','Full CAD teacher','Hidden real error [0,2]','Proxy error [0,2]')):
                                cv2.putText(header,label,(224*column+6,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
                            cv2.imwrite(str(a.out/(stem+'_teacher_error.png')),np.concatenate([header,picture],axis=0))
                f.flush()
            print(json.dumps(dict(physical_sequence=physical,rows=rows,seconds=time.monotonic()-begun)),flush=True)
    manifest.update(completed=True,rows=rows,frames_sha256=sha(a.out/'frames.jsonl'),seconds=time.monotonic()-begun)
    atomic_json(a.out/'manifest.json',manifest)
if __name__=='__main__':main()
