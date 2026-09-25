"""Train-split-only oracle readout cache; physical-sequence-disjoint holdout."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch, yaml
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lip.unified.build import build_model, make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene, encode_scenes, build_teachers
from lip.unified.serial_completion import migrate_serial, pack_completion
from lip.engine.jepa_checkpoint import sha


def physical(stream):
    return '/'.join(stream.split('|')[0].split('/')[:2])


def heldout(stream):
    return int(hashlib.sha256(physical(stream).encode()).hexdigest()[:8], 16) % 5 == 0


def initialize(config):
    plan = config['serial_completion']
    if sha(plan['source_checkpoint']) != plan['source_sha256']:
        raise ValueError('Source checkpoint changed')
    torch.manual_seed(config['seed'])
    model = build_model(config)
    source = torch.load(plan['source_checkpoint'], map_location='cpu', weights_only=False)
    if source['step'] != plan['source_step']:
        raise ValueError('Source step mismatch')
    new = migrate_serial(model, source['model'])
    model.fast_geometry = model.vector_geometry = True
    model.trusted_training_inputs = True
    return model, source, new


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8)
    p.add_argument('--records',type=int,default=24);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(0)
    c=yaml.safe_load(Path(a.config).read_text());m,source,new=initialize(c);del source
    m.requires_grad_(False).eval();factory=Factory(c,m,make_store(c,m))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);records=[];start=time.monotonic()
    counts={'train':0,'holdout':0};desired={'train':a.records,'holdout':max(4,a.records//3)}
    draw=0
    with torch.no_grad():
        while any(counts[k]<desired[k] for k in counts):
            seed=c['serial_completion']['train_seed_start']+a.rank+draw*a.world;draw+=1
            e,(truth,visible)=factory.sample(seed,frames=12)
            kind='holdout' if heldout(e.stream) else 'train'
            if counts[kind]>=desired[kind]:continue
            frame=8;gt=truth[frame]
            scene=prepare_scene(e.rgb[frame],e.depth[frame],gt,e.mesh,e.k,e.times[frame],e.stream,e.cad,factory.renderer,fast=True)
            occ=e.occlusion_plan.render(scene,frame)
            obs=encode_scenes(m,[scene],occlusions=[occ])
            target=build_teachers(m.ema_teacher,[scene],gt[None],visible[frame:frame+1],[occ.mask],factory.renderer,
                real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True)
            rendered=scene.render;mask=rendered['mask'][None,None]&scene.bounds
            xyz=rendered['xyz'][None]/scene.diameter
            # These are diagnostic ideal complete CAD observations, not student inputs.
            camera=obs.crop_rays*rendered['depth'][None]/scene.diameter-gt[None,:3,3,None,None]/scene.diameter
            packet=dict(feature=torch.cat((target.proxy_mid,target.proxy_last),-1),xyz=xyz,camera=camera,
                weight=mask.float(),measured_weight=torch.zeros_like(mask,dtype=torch.float32),completed_weight=mask.float())
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction,_=m(obs)
            valid=F.avg_pool2d(obs.packet.pixel_valid.float(),14,14).flatten(1)>=.999
            predicted=pack_completion(prediction['f_mid_predicted'],prediction['f_predicted'],
                torch.cat((prediction['surface_xyz'],prediction['surface_depth_residual'],prediction['geometry_valid_logits']),1),
                obs.geometry_image,obs.crop_rays,obs.base,obs.diameter,prediction['evidence_logits'],prediction['support_logits'],obs.mid,obs.last,valid,obs.measured_depth_m)
            cpu=lambda p:{k:v.detach().float().cpu() for k,v in p.items()}
            records.append(dict(split=kind,physical=physical(e.stream),stream=e.stream,seed=seed,frame=frame,
                truth=gt.cpu(),diameter=scene.diameter,oracle=cpu(packet),predicted=cpu(predicted)))
            counts[kind]+=1
            print(json.dumps(dict(rank=a.rank,counts=counts,seconds=time.monotonic()-start)),flush=True)
    path=out/'packets.pt';torch.save(records,path)
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,training_split='train',official_test_access=False,
        counts=counts,physical_holdout_rule='sha256 first8 modulo5 equals0',source_sha256=c['serial_completion']['source_sha256'],
        packets_sha256=sha(path),new_parameter_names=new,seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':main()
