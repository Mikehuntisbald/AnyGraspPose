"""All s0-val streams/frames; paired JEPA geometry, no pose solver or training."""
import argparse,hashlib,json,sys,time
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.unified.execution_speed import crop_images_fast
from lip.unified.renderer import FullTextureRenderer
from lip.unified.occlusion import OccluderBank,Occlusion
from lip.unified.full_recovery_metrics import recovery_metrics,visible_eval_mask
from lip.geometry.so3 import center_pose
from lip.engine.jepa_checkpoint import load_core,sha
from lip.unified.checkpoint import atomic_json

ARMS={'v52':('/mnt/why/dexycb_lip/unified_jepa_20260921/recovery_balance_v52/balanced/runs/seed42/last.pt','a6bd6204e1c646e31a02c25fd20d9cf7051354fb0e1e04b7d2f083111fc2df81'),
      'v53':('/mnt/why/dexycb_lip/unified_jepa_20260921/recovery_formal_v53/runs/seed42/last.pt','717b503de54724e7dad79fe667b9fd49fe392b898af2522267add59ee65bdb3d')}
CASES=('natural','light','heavy')


def tensor_hash(values):
    h=hashlib.sha256()
    for value in values:
        v=value.detach().cpu().contiguous().numpy();h.update(str((v.shape,str(v.dtype))).encode());h.update(v.tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--world',type=int,default=8);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();torch.cuda.set_device(0);torch.set_num_threads(2);cv2.setNumThreads(0);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    c=yaml.safe_load(Path('configs/jepa/recovery_formal_v53.yaml').read_text())
    models={}
    for arm,(checkpoint,digest) in ARMS.items():
        if sha(checkpoint)!=digest:raise ValueError('Checkpoint changed: '+arm)
        model=build_model(c);record=torch.load(checkpoint,map_location='cpu',weights_only=False)
        load_core(model,record['model']);del record
        model.requires_grad_(False).eval();model.fast_geometry=model.vector_geometry=True
        assert model.disable_history and not hasattr(model.cad_atlas_decoder,'image_readout')
        models[arm]=model
    model=models['v53'];store=make_store(c,model);renderer=FullTextureRenderer('cuda')
    index=Path(c['paths']['index_root']);raw=Path(c['paths']['data_root'])
    audit=json.loads((index/'audit.json').read_text())
    streams=sorted([json.loads(x) for x in (index/'streams.jsonl').read_text().splitlines() if json.loads(x)['split']=='val'],key=lambda x:x['stream_id'])
    assert len(streams)==320 and sum(s['num_frames'] for s in streams)==23200
    ipath=Path(c['paths']['val_initializers']);initial=json.loads(ipath.read_text())
    assert initial['completed'] and initial['split']=='val' and initial['uses_gt_pose'] is False
    assert initial['split_hash']==audit['split_hash'] and initial['mesh_hash']==audit['mesh_hash']
    initial=initial['initializers'];assert set(initial)=={s['stream_id'] for s in streams}
    reference_path=Path(c['paths']['baseline_scored'])/'predictions.jsonl'
    reference={(r['stream_id'],r['frame_index']):r for r in map(json.loads,reference_path.read_text().splitlines())}
    bank=OccluderBank(c['paths']['occluder_bank'],audit['split_hash'])
    assert bank.receipt['bank_sha256']==c['augmentation']['bank_sha256']
    selected=streams[a.rank::a.world]
    if a.smoke:selected=selected[:1]
    a.out.mkdir(parents=True,exist_ok=False);begun=time.monotonic();rows=frames=usable=0
    manifest=dict(completed=False,split='val',official_test_access=False,training=False,pose_evaluation=False,
        rank=a.rank,world=a.world,smoke=a.smoke,checkpoints={k:v[1] for k,v in ARMS.items()},
        streams=[s['stream_id'] for s in selected],index_sha256=sha(index/'streams.jsonl'),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],
        initializers_sha256=sha(ipath),reference_sha256=sha(reference_path),occluder_bank_sha256=bank.receipt['bank_sha256'],
        cases=list(CASES),student_uses_gt=False,history=False,conditional_on='same previous-frame sealed LIP pose; PoseCNN on initialization frame',
        scope='All320 validation streams and all23200 frames accounted. Geometry means require legal initialization and nonempty targets; unavailable frames and empty masks remain in coverage counts.',
        target_policy='Original raw real/proxy masks, FP32 geometry labels; no V41 training quarantine at evaluation. Visible depth uses original sensor. Both models share one target build.',
        known_limit='Shared reference-conditioned crops, not closed-loop tracking or pose evaluation; does not compare absolute numbers to controlled10deg training holdouts.')
    atomic_json(a.out/'manifest.json',manifest)
    with torch.no_grad(),(a.out/'frames.jsonl').open('w') as log:
        def emit(row):
            nonlocal rows
            log.write(json.dumps(row,allow_nan=False)+'\n');rows+=1
        for stream_index,s in enumerate(selected):
            sid=s['stream_id'];physical='/'.join(sid.split('/')[:2]);init=initial[sid]
            first=init['frame_index'] if init is not None else s['num_frames'];directory=raw/s['relative_dir']
            with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            center=torch.tensor(mesh['center'],device='cuda');d=float(mesh['diameter']);k=torch.tensor(s['intrinsics'],device='cuda')
            cad=store.get(raw/s['mesh_path'],mesh)
            assert sha(index/s['pose_cache'])==s['pose_cache_sha256']
            with np.load(index/s['pose_cache']) as z:truth=center_pose(torch.tensor(z['poses'],device='cuda'),center)
            previous=None;previous_time=None
            frame_ids=range(s['num_frames']) if not a.smoke else range(first,min(first+2,s['num_frames']))
            for frame in frame_ids:
                common=dict(stream=sid,physical_sequence=physical,object_id=s['object_id'],frame=frame)
                frames+=1
                if frame<first:
                    for case in CASES:emit(dict(**common,case=case,status='unavailable_initialization',metrics=None))
                    continue
                native_pose=init['pose_original'] if frame==first else reference[sid,frame-1]['pose_original']
                if native_pose is None:
                    for case in CASES:emit(dict(**common,case=case,status='missing_reference_pose',metrics=None))
                    previous=previous_time=None;continue
                base=center_pose(torch.tensor(native_pose,device='cuda'),center)
                image=cv2.imread(str(directory/f'color_{frame:06d}.jpg'));dep=cv2.imread(str(directory/f'aligned_depth_to_color_{frame:06d}.png'),-1)
                if image is None or dep is None:raise ValueError('Missing validation RGB-D: '+sid+'/'+str(frame))
                rgb=torch.tensor(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy(),device='cuda').float()/255
                depth=torch.tensor(dep.astype('f4')[None],device='cuda')*audit['depth_scale_to_m']
                scene=prepare_scene(rgb,depth,base,mesh,k,frame/audit['fps'],sid,cad,renderer,previous,previous_time,fast=True)
                previous=base;previous_time=frame/audit['fps']
                scenes=[replace(scene,stream=sid+'|v54|'+case) for case in CASES]
                occlusions=[Occlusion(scene.rgb,scene.depth,torch.zeros_like(scene.bounds))]
                for case,fraction in (('light',.25),('heavy',.8)):
                    seed=int(hashlib.sha256(f'v54/{sid}/{frame}/{case}'.encode()).hexdigest()[:8],16)
                    occ=bank.plan(np.random.default_rng(seed),s['object_id'],heavy=case=='heavy',start=0,duration=1,target_fraction=fraction)
                    occlusions.append(occ.render(scene,0))
                # Generate both predictions before constructing any teacher or masks.
                predictions={};raw_inputs=None
                for arm,network in models.items():
                    obs=encode_scenes(network,scenes,frame_id=frame,occlusions=occlusions)
                    current=(obs.packet.rgb_crop,obs.geometry_image,obs.measured_depth_m,obs.base,obs.state)
                    if raw_inputs is None:raw_inputs=tuple(v.detach().clone() for v in current)
                    elif not all(torch.equal(x,y) for x,y in zip(raw_inputs,current)):raise ValueError('Paired student inputs differ')
                    with torch.autocast('cuda',dtype=torch.bfloat16):prediction,_=network(obs)
                    predictions[arm]={key:prediction[key] for key in ('surface_xyz','surface_depth_m','geometry_valid_logits')}
                    del prediction,obs
                input_digest=tensor_hash(raw_inputs);del raw_inputs
                with np.load(directory/f'labels_{frame:06d}.npz') as z:native_visible=torch.tensor(z['seg']==s['object_id'],device='cuda')[None,None]
                target=build_teachers(model.ema_teacher,scenes,truth[frame:frame+1].repeat(3,1,1),native_visible.repeat(3,1,1,1),
                    [o.mask for o in occlusions],renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
                visible=(crop_images_fast(native_visible.float(),scene.affine,mode='nearest')>.5).repeat(3,1,1,1)
                added=torch.cat([o.mask for o in occlusions]);original_depth=scene.depth.repeat(3,1,1,1)
                visible_mask=visible_eval_mask(target,visible,added,original_depth,scene.bounds.repeat(3,1,1,1))
                masks=dict(real=target.geometry_real_weight,proxy=target.geometry_proxy_weight,visible=visible_mask)
                target_digest=tensor_hash([target.cad_geometry_xyz,target.surface_depth_m,*masks.values()])
                diameter=depth.new_full((3,),d);crop_k=scene.k_crop[None].repeat(3,1,1)
                scored={arm:recovery_metrics(pred,target,diameter,crop_k,masks,visible_depth_m=original_depth) for arm,pred in predictions.items()}
                positions=native_visible[0,0].nonzero();coverage=None
                if len(positions):
                    native_uv=torch.cat((positions[:,[1,0]].float(),torch.ones(len(positions),1,device='cuda')),-1)
                    crop_uv=native_uv@scene.affine.T;crop_uv=crop_uv[:,:2]/crop_uv[:,2:]
                    coverage=float(((crop_uv>=0)&(crop_uv<224)).all(-1).float().mean())
                rotation=base[:3,:3]@truth[frame,:3,:3].T
                base_rotation=float(((rotation.trace()-1)/2).clamp(-1,1).acos()*180/torch.pi)
                base_translation=float((base[:3,3]-truth[frame,:3,3]).norm()*1000)
                retained=[float((visible[i:i+1]&~o.mask).sum()/visible[i:i+1].sum()) if visible[i:i+1].any() else None for i,o in enumerate(occlusions)]
                for lane,case in enumerate(CASES):
                    emit(dict(**common,case=case,status='evaluated',input_sha256=input_digest,target_sha256=target_digest,
                        remaining_originally_visible_fraction=retained[lane],native_visibility=reference[sid,frame].get('visibility'),
                        visible_crop_coverage=coverage,base_rotation_deg=base_rotation,base_translation_mm=base_translation,
                        gt_surface_crop_pixels=int(target.cad_geometry_valid[lane].sum()),original_visible_crop_pixels=int(visible[lane].sum()),
                        metrics={arm:values[lane] for arm,values in scored.items()}))
                if stream_index==0 and frame==(first if a.smoke else min(first+24,s['num_frames']-1)):
                    np.savez_compressed(a.out/'fixed_example.npz',input_rgb=torch.cat([o.rgb for o in occlusions]).cpu().numpy(),
                        target_xyz=target.cad_geometry_xyz.cpu().numpy(),target_depth=target.surface_depth_m.cpu().numpy(),
                        real_mask=masks['real'].cpu().numpy(),proxy_mask=masks['proxy'].cpu().numpy(),diameter=d,
                        **{arm+'_'+key:value.cpu().numpy() for arm,pred in predictions.items() for key,value in pred.items()})
                usable+=1
                if usable%25==0:
                    log.flush();atomic_json(a.out/'progress.json',dict(frames=frames,evaluated_frames=usable,rows=rows,seconds=time.monotonic()-begun,stream=sid,frame=frame))
                del predictions,target,scored
            log.flush()
    manifest.update(completed=True,frames=frames,evaluated_frames=usable,records=rows,seconds=time.monotonic()-begun,
        paired_raw_inputs_exact=True,targets_built_once_per_pair=True,frames_sha256=sha(a.out/'frames.jsonl'))
    atomic_json(a.out/'manifest.json',manifest)
    print(json.dumps(dict(completed=True,rank=a.rank,frames=frames,seconds=manifest['seconds'])),flush=True)


if __name__=='__main__':main()
