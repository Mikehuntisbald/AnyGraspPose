"""V65 paired student inputs and labels; official model never reads label files."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_serial_completion import heldout
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.unified.execution_speed import crop_images_fast
from lip.unified.flow_reconstruction import flow_labels
from lip.geometry.so3 import update
from lip.engine.jepa_checkpoint import load_core,sha

SOURCE='/mnt/why/dexycb_lip/unified_jepa_20260921/local_flow_v60_r1/training/extra.pt'
DIGEST='0c6f712c7848316259dae3d24554dd4dfac506a20404676e43fb679b94d46d7e'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--split',choices=['holdout'],default='holdout');p.add_argument('--records',type=int,required=True)
    a=p.parse_args();torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    c=yaml.safe_load(Path('/mnt/why/dexycb_lip/unified_jepa_20260921/local_flow_v60_r1/training/extra.yaml').read_text());m=build_model(c)
    assert sha(SOURCE)==DIGEST
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False);load_core(m,saved['model']);del saved
    m.requires_grad_(False).eval();m.fast_geometry=m.vector_geometry=True
    f=Factory(c,m,make_store(c,m));a.out.mkdir(exist_ok=False,parents=True)
    records=[];draw=0;begun=time.monotonic()
    with torch.no_grad():
        while len(records)<a.records:
            seed=(58000000 if a.split=='train' else 65050000)+a.rank+8*draw;draw+=1
            e,(truth,mask)=f.sample(seed,frames=1)
            if heldout(e.stream)!=(a.split=='holdout'):continue
            item=len(records);d=float(e.mesh['diameter']);gt=truth[:1]
            generator=torch.Generator(device='cuda').manual_seed(seed)
            noise=torch.randn(1,6,device='cuda',generator=generator)*gt.new_tensor([.2,.2,.2,.035,.035,.035])
            base=update(gt,noise[:,:3],noise[:,3:],gt.new_tensor([d]))
            base_kind='gaussian_training'
            if a.split=='train' and item%4==3:base=e.initial[None];base_kind='transported_training_initializer'
            if a.split=='holdout':
                noise.zero_();noise[0,item%3]=((0,10,60)[(item%12)//4])*torch.pi/180*(-1 if item%2 else 1)
                base=update(gt,noise[:,:3],noise[:,3:],gt.new_tensor([d]));base_kind=f'controlled_{(0,10,60)[(item%12)//4]}'
            scene=prepare_scene(e.rgb[0],e.depth[0],base[0],e.mesh,e.k,e.times[0],e.stream,e.cad,f.renderer,fast=True)
            heavy=item%4>=2;oid=f.streams[e.stream.split('|')[0]]['object_id']
            plan=f.occluders.plan(np.random.default_rng(seed+581),oid,heavy=heavy,start=1 if item%4==0 else 0,duration=1,target_fraction=.8 if heavy else .3)
            occ=plan.render(scene,0);obs=encode_scenes(m,[scene],occlusions=[occ])
            with torch.autocast('cuda',dtype=torch.bfloat16):out,_=m(obs)
            target=build_teachers(m.ema_teacher,[scene],gt,mask,[occ.mask],f.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
            visible=(crop_images_fast(mask.float(),scene.affine,mode='nearest')>.5)&scene.bounds&~occ.mask
            labels=flow_labels(out,target,scene.k_crop[None],gt.new_tensor([d]),visible)
            target_uv=labels['uv'];reference=out['flow_reference']['uv']
            raw=occ.rgb
            assert raw.shape==(1,3,224,224) and raw.min()>=0 and raw.max()<=1
            template=scene.render['rgb'][None]
            template_mask=scene.render['mask'][None].bool()
            if template_mask.ndim==4:template_mask=template_mask[:,0]
            assert template_mask.shape==(1,224,224)
            input_file=a.out/f'input{item:03d}.pt'
            torch.save(dict(query=raw.cpu(),template=template.cpu(),mask=template_mask.cpu()),input_file)
            torch.save(dict(reference=reference.cpu(),truth=target_uv.cpu(),regions={k:labels[k].cpu() for k in ('observed','real','proxy')},
                baseline=out['flow_rounds'][-1]['uv'].cpu()),a.out/f'labels{item:03d}.pt')
            row=dict(seed=seed,stream=e.stream,heavy=heavy,base_kind=base_kind,input_file=input_file.name,input_sha256=sha(input_file))
            records.append(row)
            if len(records)%32==0:print(json.dumps(dict(records=len(records),seconds=time.monotonic()-begun)),flush=True)
    (a.out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (a.out/'receipt.json').write_text(json.dumps(dict(completed=True,source_sha256=DIGEST,
        rank=a.rank,records=len(records),training_partition_only=True,physical_holdout=True,
        teacher_in_student=False,weights_frozen=True,default_model_changed=False,
        scope='Controlled0/10/60 reference, actual corrupted student RGB, estimated-pose render only. Separate immutable label files. No oracle image overlay in student.',
        seconds=time.monotonic()-begun),indent=2)+'\n')


if __name__=='__main__':main()
