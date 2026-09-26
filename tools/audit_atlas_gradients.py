"""Decompose initial V42 geometry gradients on the actual first paired batch."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_serial_completion import heldout
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.unified.execution_speed import crop_images_fast
from lip.unified.paired_geometry_curriculum import paired_scenes
from lip.unified.supervision_quality import quarantine_real_geometry
from lip.unified.geometry_transport_objective import objective
from lip.unified.cad_atlas_decoder import atlas_correspondence_loss
from lip.unified.surface_normals import normal_terms
from lip.unified.recovery_focus import masked_mean


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    c=yaml.safe_load(Path(a.config).read_text());m=build_model(c)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);del ck
    m.fast_geometry=m.vector_geometry=m.trusted_training_inputs=True
    for n,t in m.named_parameters():t.requires_grad_(not n.startswith('ema_teacher.'))
    factory=Factory(c,m,make_store(c,m));episodes=[];targets=[];seeds=[]
    for lane in range(4):
        for attempt in range(100):
            seed=40000000+lane+attempt*100000003;e,t=factory.sample(seed,frames=1)
            if not heldout(e.stream):break
        episodes.append(e);targets.append(t);seeds.append(seed)
    gt=torch.stack([t[0][0] for t in targets]);masks=torch.stack([t[1][0] for t in targets])
    scenes=[prepare_scene(e.rgb[0],e.depth[0],gt[i],e.mesh,e.k,e.times[0],e.stream,e.cad,factory.renderer,fast=True) for i,e in enumerate(episodes)]
    occlusions=[]
    for e,s,seed in zip(episodes,scenes,seeds):
        rng=np.random.default_rng(seed+34001);kind=int(rng.integers(4));oid=factory.streams[e.stream.split('|')[0]]['object_id']
        occlusions.append(factory.occluders.plan(rng,oid,heavy=kind>=2,start=0 if kind else 1,duration=1).render(s,0))
    scenes+=paired_scenes(scenes,gt,factory.renderer);occlusions+=occlusions.copy();gt=gt.repeat(2,1,1);masks=masks.repeat(2,1,1,1)
    obs=encode_scenes(m,scenes,occlusions=occlusions)
    target=build_teachers(m.ema_teacher,scenes,gt,masks,[o.mask for o in occlusions],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
    visible=torch.cat([(crop_images_fast(masks[i:i+1].float(),s.affine,mode='nearest')>.5)&s.bounds for i,s in enumerate(scenes)])
    target=quarantine_real_geometry(target,scenes,visible,.03);visible=visible&~torch.cat([o.mask for o in occlusions])
    with torch.autocast('cuda',dtype=torch.bfloat16):out,_=m(obs)
    normal_weight=c.get('cad_atlas',{}).get('normal_weight',.02)
    full,_=objective(out,target,obs,visible,torch.stack([s.k_crop for s in scenes]),False,canonical_surface=True,normal_weight=normal_weight,
                     fallback_xyz_weight=c.get('cad_atlas',{}).get('fallback_xyz_weight',0.))
    normal=out['surface_xyz'].sum()*0
    for mask,factor in [(target.geometry_real_weight,1.),(target.geometry_proxy_weight,.5)]:
        error,valid,_=normal_terms(out['surface_xyz'],target.cad_geometry_xyz,mask,2)
        normal=normal+normal_weight*factor*masked_mean(error,valid)
    ce,_=atlas_correspondence_loss(out,target,visible&target.real_geometry_eligible)
    names=['cad_atlas_decoder.query.0.weight','cad_atlas_decoder.descriptor.1.weight']
    parameters=[dict(m.named_parameters())[n] for n in names]
    gradients={}
    for name,loss in [('full_geometry',full),('normal',normal),('correspondence',ce)]:
        gradients[name]=torch.cat([g.flatten() for g in torch.autograd.grad(loss,parameters,retain_graph=True)]).detach().float()
    gradients['other_geometry']=gradients['full_geometry']-gradients['normal']
    fallback_gradient,=torch.autograd.grad(full+ce,out['atlas_fallback'],retain_graph=True)
    result=dict(completed=True,training=False,optimizer_updates=0,seeds=seeds,pose_hypotheses=8,
        losses=dict(full_geometry=float(full.detach()),normal=float(normal.detach()),correspondence=float(ce.detach())),
        gradient_norms={k:float(g.norm()) for k,g in gradients.items()},
        normal_to_ce_norm_ratio=float(gradients['normal'].norm()/gradients['correspondence'].norm()),
        final_fallback_xyz_gradient_norm=float(fallback_gradient[:,:3].norm()),
        final_fallback_depth_validity_gradient_norm=float(fallback_gradient[:,3:].norm()),
        checkpoint=a.checkpoint,normal_weight=normal_weight,
        scope='Fixed V42 first rank0 paired batch and explicitly supplied checkpoint; norms concatenated over atlas query0 and descriptor1 weights')
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
