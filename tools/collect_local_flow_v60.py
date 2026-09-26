"""V60 fixed local-flow features with disjoint physical holdout."""
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

SOURCE='/mnt/why/dexycb_lip/unified_jepa_20260921/flow_reconstruction_v56/runs/seed42/last.pt'
DIGEST='044a8f928d324e54c8e2cab2927dd31e55942cde98bc09fa5f18ce5245d0fa5f'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--split',choices=['train','holdout'],required=True);p.add_argument('--records',type=int,required=True)
    a=p.parse_args();torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    c=yaml.safe_load(Path('configs/jepa/flow_reconstruction_v56.yaml').read_text());m=build_model(c)
    assert sha(SOURCE)==DIGEST
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False);load_core(m,saved['model']);del saved
    m.requires_grad_(False).eval();m.fast_geometry=m.vector_geometry=True;m.flow_reconstruction.capture_local_flow=True
    f=Factory(c,m,make_store(c,m));a.out.mkdir(exist_ok=False,parents=True)
    features=[];labels=[];groups=[];coarse=[];regions=[];records=[];draw=0;begun=time.monotonic()
    with torch.no_grad():
        while len(records)<a.records:
            seed=(60000000 if a.split=='train' else 59050000)+a.rank+8*draw;draw+=1
            e,(truth,mask)=f.sample(seed,frames=1)
            if heldout(e.stream)!=(a.split=='holdout'):continue
            item=len(records);d=float(e.mesh['diameter']);gt=truth[:1]
            generator=torch.Generator(device='cuda').manual_seed(seed)
            noise=torch.randn(1,6,device='cuda',generator=generator)*gt.new_tensor([.2,.2,.2,.035,.035,.035])
            base=update(gt,noise[:,:3],noise[:,3:],gt.new_tensor([d]))
            base_kind='gaussian_training'
            if a.split=='train' and item%4==3:base=e.initial[None];base_kind='transported_training_initializer'
            if a.split=='holdout':
                noise.zero_();noise[0,item%3]=(10 if item%8<4 else 60)*torch.pi/180*(-1 if item%2 else 1)
                base=update(gt,noise[:,:3],noise[:,3:],gt.new_tensor([d]));base_kind=f'controlled_{10 if item%8<4 else 60}'
            scene=prepare_scene(e.rgb[0],e.depth[0],base[0],e.mesh,e.k,e.times[0],e.stream,e.cad,f.renderer,fast=True)
            heavy=item%4>=2;oid=f.streams[e.stream.split('|')[0]]['object_id']
            plan=f.occluders.plan(np.random.default_rng(seed+581),oid,heavy=heavy,start=1 if item%4==0 else 0,duration=1,target_fraction=.8 if heavy else .3)
            occ=plan.render(scene,0);obs=encode_scenes(m,[scene],occlusions=[occ])
            with torch.autocast('cuda',dtype=torch.bfloat16):out,_=m(obs)
            target=build_teachers(m.ema_teacher,[scene],gt,mask,[occ.mask],f.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
            visible=(crop_images_fast(mask.float(),scene.affine,mode='nearest')>.5)&scene.bounds&~occ.mask
            value=flow_labels(out,target,scene.k_crop[None],gt.new_tensor([d]),visible)
            masks=torch.stack([value[k] for k in ('observed','real','proxy')],-1)
            known=masks.any(-1)
            for result in out['flow_rounds']:
                x=result['local_flow_features'][known].float().cpu()
                features.append(x);labels.append(value['uv'][known].cpu())
                coarse.append(result['coarse_uv'][known].cpu());regions.append(masks[known].cpu())
                groups.append(torch.full((len(x),),item,dtype=torch.long))
            records.append(dict(seed=seed,stream=e.stream,heavy=heavy,base_kind=base_kind,points=int(known.sum())))
            if len(records)%32==0:print(json.dumps(dict(records=len(records),seconds=time.monotonic()-begun)),flush=True)
    torch.save(dict(features=torch.cat(features),labels=torch.cat(labels),groups=torch.cat(groups),coarse=torch.cat(coarse),regions=torch.cat(regions),records=records),a.out/'features.pt')
    (a.out/'receipt.json').write_text(json.dumps(dict(completed=True,source_sha256=DIGEST,features_sha256=sha(a.out/'features.pt'),
        rank=a.rank,records=len(records),points=sum(len(x) for x in features),split=a.split,training_partition_only=True,
        official_test_access=False,teacher_in_student=False,backbone_frozen=True,feature_precision='float32',
        physical_sequences=sorted({'/'.join(r['stream'].split('|')[0].split('/')[:2]) for r in records}),
        target_definition='Exact GT projection of CAD sample; observed/artificial-real-hidden/CAD-proxy ownership from audited flow_labels. Both original frozen rounds cached. No teacher input.',
        seconds=time.monotonic()-begun),indent=2)+'\n')


if __name__=='__main__':main()
