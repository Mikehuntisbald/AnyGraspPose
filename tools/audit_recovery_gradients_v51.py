"""Measure competing recovery gradients on distinct actual training batches; no updates."""
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
from lip.geometry.so3 import update
from torch.nn import functional as F
from lip.engine.jepa_checkpoint import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42+a.rank)
    c=yaml.safe_load(Path(a.config).read_text());m=build_model(c)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);del ck
    m.fast_geometry=m.vector_geometry=m.trusted_training_inputs=True
    for n,t in m.named_parameters():t.requires_grad_(not n.startswith('ema_teacher.'))
    factory=Factory(c,m,make_store(c,m));episodes=[];targets=[];seeds=[]
    for lane in range(4):
        for attempt in range(100):
            seed=51000000+a.rank*4+lane+attempt*100000003;e,t=factory.sample(seed,frames=1)
            if not heldout(e.stream):break
        episodes.append(e);targets.append(t);seeds.append(seed)
    gt=torch.stack([t[0][0] for t in targets]);masks=torch.stack([t[1][0] for t in targets])
    diameter=gt.new_tensor([float(e.mesh['diameter']) for e in episodes])
    noise=torch.randn(4,6,device='cuda')*gt.new_tensor([.2]*3+[.035]*3)
    base=update(gt,noise[:,:3],noise[:,3:],diameter)
    scenes=[prepare_scene(e.rgb[0],e.depth[0],base[i],e.mesh,e.k,e.times[0],e.stream,e.cad,factory.renderer,fast=True) for i,e in enumerate(episodes)]
    occlusions=[]
    for e,s,seed in zip(episodes,scenes,seeds):
        rng=np.random.default_rng(seed+34001);kind=int(rng.integers(4));oid=factory.streams[e.stream.split('|')[0]]['object_id']
        occlusions.append(factory.occluders.plan(rng,oid,heavy=kind>=2,start=0 if kind else 1,duration=1).render(s,0))
    scenes+=paired_scenes(scenes,gt,factory.renderer);occlusions+=occlusions.copy();gt=gt.repeat(2,1,1);masks=masks.repeat(2,1,1,1)
    obs=encode_scenes(m,scenes,occlusions=occlusions)
    target=build_teachers(m.ema_teacher,scenes,gt,masks,[o.mask for o in occlusions],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
    visible=torch.cat([(crop_images_fast(masks[i:i+1].float(),s.affine,mode='nearest')>.5)&s.bounds for i,s in enumerate(scenes)])
    target=quarantine_real_geometry(target,scenes,visible,.03);visible=visible&~torch.cat([o.mask for o in occlusions])
    captured={}
    hook=m.surface_head.output[-1].register_forward_pre_hook(lambda _,args: captured.update(dense=args[0]))
    with torch.autocast('cuda',dtype=torch.bfloat16):out,_=m(obs)
    hook.remove()
    normal_weight=c.get('cad_atlas',{}).get('normal_weight',.02)
    full,_=objective(out,target,obs,visible,torch.stack([s.k_crop for s in scenes]),False,canonical_surface=True,normal_weight=normal_weight,
                     fallback_xyz_weight=c.get('cad_atlas',{}).get('fallback_xyz_weight',0.))
    normal=out['surface_xyz'].sum()*0
    for mask,factor in [(target.geometry_real_weight,1.),(target.geometry_proxy_weight,.5)]:
        error,valid,_=normal_terms(out['surface_xyz'],target.cad_geometry_xyz,mask,2)
        normal=normal+normal_weight*factor*masked_mean(error,valid)
    ce,_=atlas_correspondence_loss(out,target,visible&target.real_geometry_eligible)
    # Isolate exactly the dense/coarse depth-regression terms from objective().
    depth=out['surface_depth_residual'].sum()*0
    for prediction,factor in [(out['surface_depth_residual'],1.),(out['coarse_surface'][:,3:4],.2)]:
        error=F.smooth_l1_loss(prediction.float(),target.surface_depth_residual.float(),beta=.02,reduction='none')
        for mask,weight in [(target.geometry_real_weight,1.),(target.geometry_proxy_weight,.5)]:
            depth=depth+factor*weight*masked_mean(error,mask)
    tensors={'patch':out['patch_latent'],'dense':captured['dense'],
        'dpt_shared':m.surface_head.refine[-1].project.weight,
        'atlas_query':m.cad_atlas_decoder.query[0].weight,
        'dpt_output':m.surface_head.output[-1].weight}
    gradients={}
    losses={'geometry':full,'depth':depth,'correspondence':ce}
    for name,loss in losses.items():
        gradients[name]={k:(g.detach().float().flatten() if g is not None else torch.zeros_like(t).float().flatten())
            for (k,t),g in zip(tensors.items(),torch.autograd.grad(loss,list(tensors.values()),retain_graph=True,allow_unused=True))}
    gradients['other_geometry']={k:gradients['geometry'][k]-gradients['depth'][k] for k in tensors}
    stats={}
    for k in tensors:
        g,ceg,d=gradients['geometry'][k],gradients['correspondence'][k],gradients['depth'][k]
        stats[k]=dict(norms={n:float(v[k].norm()) for n,v in gradients.items()},
            geometry_ce_cosine=float(F.cosine_similarity(g[None],ceg[None])),
            depth_ce_cosine=float(F.cosine_similarity(d[None],ceg[None])),
            combined_depth_cosine=float(F.cosine_similarity((g+ceg)[None],d[None])))
    coverage={}
    for name,mask in [('real',target.geometry_real_weight),('proxy',target.geometry_proxy_weight),('visible',visible&target.real_geometry_eligible)]:
        mask=mask&target.cad_geometry_valid
        counts=mask.flatten(1).sum(1)
        coverage[name]=dict(eligible_pixels=counts.tolist(),ce_pixels=counts.clamp_max(256).tolist())
    result=dict(completed=True,training=False,optimizer_updates=0,rank=a.rank,seeds=seeds,pose_hypotheses=8,
        losses={k:float(v.detach()) for k,v in losses.items()},gradients=stats,coverage=coverage,
        checkpoint=a.checkpoint,checkpoint_sha256=sha(a.checkpoint),normal_weight=normal_weight,
        scope='Distinct training-only32 observations /64 hypotheses across8 ranks; actual canonical masks; no optimizer or pose metrics. Gradient cosine is diagnostic, not evidence of causal efficacy.')
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
