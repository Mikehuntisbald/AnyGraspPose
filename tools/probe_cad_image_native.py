"""Fixed40 sequences /120 native frames, natural and heavy inputs.

Previous sealed LIP poses set every model's crop/base. This is conditional
one-step evaluation, not a closed-loop native tracking score. GT is labels only.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.unified.execution_speed import crop_images_fast
from lip.geometry.so3 import center_pose
from lip.engine.jepa_checkpoint import load_core,sha
from cad_image_diagnostics import diagnose


def main():
    p=argparse.ArgumentParser()
    for key in ('config','checkpoint','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8)
    a=p.parse_args();torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(42)
    c=yaml.safe_load(Path(a.config).read_text());model=build_model(c)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);load_core(model,saved['model']);del saved
    model.requires_grad_(False).eval();model.fast_geometry=model.vector_geometry=True
    factory=Factory(c,model,make_store(c,model),split='val')
    root=Path(c['paths']['data_root']);index=Path(c['paths']['index_root'])
    streams={x['stream_id']:x for x in map(json.loads,(index/'streams.jsonl').read_text().splitlines()) if x['split']=='val'}
    init=json.loads(Path(c['paths']['val_initializers']).read_text())['initializers']
    reference_path=Path(c['paths']['baseline_scored'])/'predictions.jsonl'
    reference={(x['stream_id'],x['frame_index']):x for x in map(json.loads,reference_path.read_text().splitlines())}
    chosen={}
    for sid,s in sorted(streams.items()):
        if init[sid] is not None and s['num_frames']-init[sid]['frame_index']>=40:chosen.setdefault('/'.join(sid.split('/')[:2]),sid)
    assert len(chosen)==40
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);count=0;begun=time.monotonic()
    with torch.no_grad(),(out/'frames.jsonl').open('w') as writer:
        for physical,sid in sorted(chosen.items())[a.rank::a.world]:
            s=streams[sid];first=init[sid]['frame_index'];directory=root/s['relative_dir']
            with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            center=torch.tensor(mesh['center'],device='cuda');d=float(mesh['diameter']);k=torch.tensor(s['intrinsics'],device='cuda')
            cad=factory.store.get(root/s['mesh_path'],mesh)
            with np.load(index/s['pose_cache']) as z:truth=center_pose(torch.tensor(z['poses'],device='cuda'),center)
            worst=min(range(first+8,s['num_frames']),key=lambda f:2. if reference[sid,f]['visibility'] is None else reference[sid,f]['visibility'])
            frames=sorted(set((first+8,min(first+24,s['num_frames']-1),worst)))
            for frame in frames:
                im=cv2.imread(str(directory/f'color_{frame:06d}.jpg'));dep=cv2.imread(str(directory/f'aligned_depth_to_color_{frame:06d}.png'),-1)
                rgb=torch.tensor(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy(),device='cuda').float()/255
                depth=torch.tensor(dep.astype('f4')[None],device='cuda')*factory.audit['depth_scale_to_m']
                with np.load(directory/f'labels_{frame:06d}.npz') as z:mask=torch.tensor(z['seg']==s['object_id'],device='cuda')[None,None]
                old=reference[sid,frame-1]['pose_original']
                if old is None:raise ValueError('Missing previous native baseline pose')
                base=center_pose(torch.tensor(old,device='cuda'),center)
                scene=prepare_scene(rgb,depth,base,mesh,k,frame/factory.audit['fps'],sid,cad,factory.renderer,fast=True)
                for heavy in (False,True):
                    seed=int(hashlib.sha256(f'{sid}/{frame}/{int(heavy)}'.encode()).hexdigest()[:8],16)
                    plan=factory.occluders.plan(np.random.default_rng(seed),s['object_id'],heavy=heavy,start=0 if heavy else 1,duration=1,target_fraction=.8 if heavy else 0.)
                    occ=plan.render(scene,0)
                    obs=encode_scenes(model,[scene],occlusions=[occ])
                    with torch.autocast('cuda',dtype=torch.bfloat16):output,_=model(obs)
                    target=build_teachers(model.ema_teacher,[scene],truth[frame:frame+1],mask,[occ.mask],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
                    visible=(crop_images_fast(mask.float(),scene.affine,mode='nearest')>.5)&scene.bounds&~occ.mask
                    diagnostics=diagnose(output,target,scene,obs,visible,truth[frame],mesh,seed,factory.renderer,model)
                    metrics={}
                    for name,region in [('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight)]:
                        metrics[name]=dict(pixels=int(region.sum()),canonical_xyz_mm=float((output['surface_xyz']-target.cad_geometry_xyz).norm(dim=1,keepdim=True)[region].mean()*d*1000),
                            depth_mm=float((output['surface_depth_m']-target.surface_depth_m).abs()[region].mean()*1000)) if region.any() else None
                    row=dict(seed=seed,stream=sid,physical_sequence=physical,frame_index=frame,heavy=heavy,window=dict(frame=frame),metrics=metrics,correspondence=diagnostics)
                    writer.write(json.dumps(row)+'\n');writer.flush();count+=1
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,records=count,checkpoint_sha256=sha(a.checkpoint),baseline_sha256=sha(reference_path),
        physical_sequences=40,selection='lexicographic stream per physical sequence; first+8, first+24, baseline lowest visibility; duplicates removed',
        scope='native previous-baseline pose/crop, conditional single-frame refinement, natural and heavy; not closed-loop',
        gt_input=False,official_test=False,seconds=time.monotonic()-begun),indent=2))


if __name__=='__main__':main()
