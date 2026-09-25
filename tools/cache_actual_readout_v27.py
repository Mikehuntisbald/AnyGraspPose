"""Actual-prediction readout cache: fixed backbone, no teacher forward/geometry."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from dataclasses import replace
import numpy as np,torch,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes
from lip.unified.execution_speed import crop_images_fast
from lip.geometry.so3 import update
from lip.engine.jepa_checkpoint import load_core,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.set_device(0)
    c=yaml.safe_load(Path(a.config).read_text());m=build_model(c);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);load_core(m,saved['model']);del saved
    m.requires_grad_(False).eval();m.fast_geometry=m.vector_geometry=True
    factory=Factory(c,m,make_store(c,m));info=json.loads(Path(c['cad_surface']['models_info']).read_text())
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False);records=[];counts=dict(train=0,heldout=0);needed=dict(train=24,heldout=8);captured={};start=time.monotonic()
    def capture(name):
        def hook(module,args):captured[name]=args[0].detach()
        return hook
    hooks=[m.geometry_readout.appearance.register_forward_pre_hook(capture('feature_observed_visible')),m.geometry_readout.relation.register_forward_pre_hook(capture('pooled')),m.geometry_readout.moments.register_forward_pre_hook(capture('moments'))]
    draw=0
    with torch.no_grad():
        while any(counts[k]<needed[k] for k in counts):
            ordinal=a.rank+draw*a.world;seed=21000000+ordinal;draw+=1
            e,(truth,visible)=factory.sample(seed,frames=12);physical='/'.join(e.stream.split('|')[0].split('/')[:2]);kind='heldout' if int(hashlib.sha256(physical.encode()).hexdigest()[:8],16)%5==0 else 'train'
            if counts[kind]>=needed[kind]:continue
            frame=8;gt=truth[frame:frame+1];d=float(e.mesh['diameter']);oid=factory.streams[e.stream.split('|')[0]]['object_id'];heavy=ordinal%2==0
            rng=np.random.default_rng(seed);noise=torch.tensor(rng.normal(size=(1,6)),device='cuda',dtype=torch.float32)*gt.new_tensor([.15]*3+[.035]*3)
            reference=update(gt,noise[:,:3],noise[:,3:],gt.new_tensor([d]))[0]
            scene=prepare_scene(e.rgb[frame],e.depth[frame],reference,e.mesh,e.k,e.times[frame],e.stream,e.cad,factory.renderer,fast=True)
            e.occlusion_plan=factory.occluders.plan(rng,oid,heavy=True,start=4,duration=8,target_fraction=.8)
            occ=e.occlusion_plan.render(scene,frame) if heavy else None
            vm=crop_images_fast(visible[frame:frame+1].float(),scene.affine,mode='nearest')>.5
            fraction=float((vm&occ.mask).sum()/vm.sum().clamp_min(1)) if occ is not None else 0.
            axis=(ordinal//2)%3;cases=[('zero',gt[0])]
            for sign in (1,-1):
                delta=gt.new_zeros(1,6);delta[0,axis]=sign*torch.pi/18
                cases.append(('positive' if sign==1 else 'negative',update(gt,delta[:,:3],delta[:,3:],gt.new_tensor([d]))[0]))
            cases.append(('mixed',reference));first=None
            for case,base in cases:
                state=scene.state.clone();state[:6]=base[:3,:2].T.flatten();state[6:9]=base[:3,3]/d
                current=replace(scene,pose=base,state=state,render=factory.renderer(e.cad['appearance'],base,scene.k_crop,224))
                obs=encode_scenes(m,[current],occlusions=[occ] if occ is not None else None)
                if first is None:first=(obs.packet.rgb_crop.clone(),obs.measured_depth_m.clone())
                assert torch.equal(first[0],obs.packet.rgb_crop) and torch.equal(first[1],obs.measured_depth_m)
                captured.clear()
                with torch.autocast('cuda',dtype=torch.bfloat16):output,_=m(obs)
                cpu=lambda v:v.detach().float().cpu()
                row={k:cpu(v) for k,v in captured.items()}
                row.update(feature_decoded=cpu(torch.cat((output['f_mid_predicted'],output['f_predicted']),-1)),feature_patch=cpu(output['patch_latent']),
                    token_valid=output['serial_token_valid'].cpu(),scale=cpu(output['serial_residual_scale']),base=base[None].cpu(),truth=gt.cpu(),diameter=torch.tensor([d]),
                    original_delta=cpu(torch.cat((output['delta_rotvec'],output['delta_center_norm']),-1)),physical=physical,stream=e.stream,split=kind,seed=seed,case=case,axis=axis,heavy=heavy,added_visible_fraction=fraction,
                    symmetric=bool(info[str(oid)].get('symmetries_discrete') or info[str(oid)].get('symmetries_continuous')))
                records.append(row)
            counts[kind]+=1
            print(json.dumps(dict(rank=a.rank,groups=counts,rows=len(records),seconds=time.monotonic()-start)),flush=True)
    for h in hooks:h.remove()
    torch.save(records,root/'records.pt')
    receipt=dict(completed=True,groups=counts,rows=len(records),checkpoint_sha256=sha(a.checkpoint),cache_sha256=sha(root/'records.pt'),teacher_forward=False,teacher_geometry_input=False,optimizer_updates=0,training_split_only=True,paired_rgbd_exact=True,scope='Controlled training-split readout probe; estimated-pose augmentation uses GT only to construct supervised training cases')
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
