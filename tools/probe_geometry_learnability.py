"""Frozen-input, train-partition-only overfit and loss-interference diagnostic.

Cache paired estimated poses of the SAME observed crop. Compare flow-only and
the existing geometry objective on one/eight records. These heads are never
installed into a production checkpoint. Labels are never decoder inputs.
"""
import argparse,json,sys,time,random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.cad_transport import CADTransport,transport_targets
from lip.unified.geometry_transport_objective import objective
from lip.unified.recovery_focus import masked_mean
from lip.engine.jepa_checkpoint import sha

SOURCE='/mnt/why/dexycb_lip/unified_jepa_20260921/geometry_transport_conditioned_v36/runs/seed42/initial.pt'
SOURCE_SHA='6bf4c1b03e68c8814373b2ce6834574cd3476e7053cf552df7759d4923edd6cc'
TARGET_KEYS=('surface_xyz','surface_depth_residual','geometry_weight','geometry_real_weight','geometry_proxy_weight',
             'geometry_valid_label','visible_label','support_label','camera_rotation','camera_translation_d','camera_rays','base_depth_d')


def cache(a):
    from prepare_serial_completion import heldout,physical
    from lip.unified.build import build_model,make_store
    from lip.unified.training import Factory
    from lip.unified.features import prepare_scene,encode_scenes,build_teachers
    from lip.unified.execution_speed import crop_images_fast
    from lip.engine.jepa_checkpoint import load_core
    from lip.geometry.so3 import update
    assert sha(SOURCE)==SOURCE_SHA
    c=yaml.safe_load(Path(a.config).read_text());m=build_model(c)
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False);load_core(m,saved['model']);del saved
    m.requires_grad_(False).eval();m.fast_geometry=m.vector_geometry=True
    factory=Factory(c,m,make_store(c,m));rows=[];captures=[];attempts={}
    hook=m.cad_transport.register_forward_pre_hook(lambda module,args:captures.append(args))
    cpu=lambda x:x.detach().cpu()
    with torch.no_grad():
        for split in ('train','holdout'):
            for draw in range(1000):
                seed=47000000+(100000 if split=='holdout' else 0)+a.rank+draw*8
                ep,(poses,masks)=factory.sample(seed,frames=1)
                if heldout(ep.stream)!=(split=='holdout'):continue
                gt=poses[:1];diam=gt.new_tensor([float(ep.mesh['diameter'])]);noise=gt.new_zeros(1,6);noise[0,a.rank%3]=torch.pi/18
                bases=[update(gt,sign*noise[:,:3],noise[:,3:],diam)[0] for sign in (1,-1)]
                first=prepare_scene(ep.rgb[0],ep.depth[0],bases[0],ep.mesh,ep.k,ep.times[0],ep.stream,ep.cad,factory.renderer,fast=True)
                oid=factory.streams[ep.stream.split('|')[0]]['object_id']
                plan=factory.occluders.plan(np.random.default_rng(seed+37),oid,heavy=True,start=0,duration=1,target_fraction=.8)
                occ=plan.render(first,0)
                original=[];pair=[]
                for sign,base in zip((1,-1),bases):
                    state=first.state.clone();state[:6]=base[:3,:2].T.flatten();state[6:9]=base[:3,3]/first.diameter
                    scene=replace(first,pose=base,state=state,render=factory.renderer(first.cad['appearance'],base,first.k_crop,224))
                    obs=encode_scenes(m,[scene],occlusions=[occ])
                    target=build_teachers(m.ema_teacher,[scene],gt,masks,[occ.mask],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True,geometry_only=True)
                    visible=(crop_images_fast(masks.float(),scene.affine,mode='nearest')>.5)&~occ.mask&scene.bounds
                    measured=obs.measured_depth_m.float()
                    camera=target.camera_rays*measured/diam[:,None,None,None]-target.camera_translation_d[:,:,None,None]
                    vx=torch.einsum('bji,bjhw->bihw',target.camera_rotation,camera)
                    vm=visible&(measured>0)&torch.isfinite(measured)&(vx.square().sum(1,keepdim=True)<1.)
                    correspondence=transport_targets(torch.where(vm,vx,target.surface_xyz),obs.geometry_image,obs.base,diam,scene.k_crop[None],obs.cad_valid)
                    flow_mask=correspondence['supported']&(target.geometry_weight|vm)
                    if int(flow_mask.sum())<128 or int(target.geometry_weight.sum())<128:break
                    captures.clear()
                    with torch.autocast('cuda',dtype=torch.bfloat16):output,_=m(obs)
                    assert len(captures)==1
                    dense,fallback,geometry,cad_valid=captures[0]
                    pair.append(dict(split=split,physical=physical(ep.stream),stream=ep.stream,seed=seed,sign=sign,
                        inputs={k:cpu(v) for k,v in dict(dense=dense,fallback=fallback,geometry=geometry,cad_valid=cad_valid).items()},
                        observation={k:cpu(getattr(obs,k)) for k in ('measured_depth_m','diameter','base','geometry_image','cad_valid')},
                        target={k:cpu(getattr(target,k)) for k in TARGET_KEYS},visible=cpu(visible),crop_k=cpu(scene.k_crop[None]),
                        flow=cpu(correspondence['flow']),flow_mask=cpu(flow_mask),patch=cpu(output['patch_latent']),
                        evidence=cpu(output['evidence_logits']),support=cpu(output['support_logits'])))
                    original.append(cpu(obs.packet.rgb_crop))
                if len(pair)==2:
                    assert torch.equal(original[0],original[1])
                    assert torch.equal(pair[0]['observation']['measured_depth_m'],pair[1]['observation']['measured_depth_m'])
                    assert torch.equal(pair[0]['target']['surface_xyz'],pair[1]['target']['surface_xyz'])
                    rows+=pair;attempts[split]=draw+1;break
            else:raise RuntimeError('No eligible paired crop')
    hook.remove();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.save(rows,out/'cache.pt')
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,source_sha256=SOURCE_SHA,records=len(rows),attempts=attempts,
        cache_sha256=sha(out/'cache.pt'),teacher_inputs=False,paired_rgbd_exact=True,training_partition_only=True,
        official_test_access=False,filter='>=128 eligible flow and missing geometry pixels; no model-error filtering'),indent=2)+'\n')


def batch(rows):
    cat=lambda xs:torch.cat(xs).cuda()
    b={k:{name:cat([r[k][name] for r in rows]) for name in rows[0][k]} for k in ('inputs','observation','target')}
    for k in ('visible','crop_k','flow','flow_mask','evidence','support'):b[k]=cat([r[k] for r in rows])
    return b


def forward(model,b):
    with torch.autocast('cuda',dtype=torch.bfloat16):surface,extra=model(**b['inputs'])
    out=dict(surface_xyz=surface[:,:3],surface_depth_residual=surface[:,3:4],geometry_valid_logits=surface[:,4:5],
             evidence_logits=b['evidence'],support_logits=b['support'],**extra)
    flow=masked_mean(F.smooth_l1_loss(out['transport_flow']/224.,b['flow']/224.,beta=.02,reduction='none').mean(1,keepdim=True),b['flow_mask'])
    mixed,metrics=objective(out,SimpleNamespace(**b['target']),SimpleNamespace(**b['observation']),b['visible'],b['crop_k'],True)
    epe=masked_mean((out['transport_flow']-b['flow']).norm(dim=1,keepdim=True),b['flow_mask'])
    return flow,mixed,epe,metrics


def fit(a):
    records=[]
    for path in sorted(Path(a.cache).glob('rank*/cache.pt')):
        receipt=json.loads(path.with_name('receipt.json').read_text());assert receipt['cache_sha256']==sha(path)
        records+=torch.load(path,weights_only=False)
    train=[r for r in records if r['split']=='train'][:a.count];test=[r for r in records if r['split']=='holdout']
    assert not ({r['physical'] for r in train}&{r['physical'] for r in test})
    model=CADTransport(reference_conditioned=True).cuda()
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
    model.load_state_dict({k.removeprefix('cad_transport.'):v for k,v in saved['model'].items() if k.startswith('cad_transport.')});del saved
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01,fused=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);resident=batch(train);held=batch(test);started=time.monotonic()
    measurements=[]
    @torch.no_grad()
    def evaluate(b):
        flow,mixed,epe,metrics=forward(model,b)
        zero=masked_mean(b['flow'].norm(dim=1,keepdim=True),b['flow_mask'])
        return dict(epe=float(epe),zero_epe=float(zero),flow_loss=float(flow),mixed_loss=float(mixed),**{k:float(v) for k,v in metrics.items()})
    def record(step):
        row=dict(step=step,train=evaluate(resident),holdout=evaluate(held),seconds=time.monotonic()-started)
        measurements.append(row);print(json.dumps(row),flush=True)
    record(0)
    flow,mixed,_,_=forward(model,resident)
    params=list(model.parameters())
    gf=torch.autograd.grad(.25*flow,params,retain_graph=True);go=torch.autograd.grad(mixed-.25*flow,params)
    f=torch.cat([x.flatten() for x in gf]).float();g=torch.cat([x.flatten() for x in go]).float()
    gradient=dict(flow_norm=float(f.norm()),other_norm=float(g.norm()),cosine=float(f@g/(f.norm()*g.norm()).clamp_min(1e-12)))
    del f,g,gf,go,flow,mixed
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        flow,mixed,_,_=forward(model,resident)
        loss=.25*flow if a.loss=='flow' else mixed
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if (step+1)%100==0 or step+1==a.steps:record(step+1)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),step=a.steps),out/'diagnostic_head.pt')
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,source_sha256=SOURCE_SHA,loss=a.loss,training_records=len(train),holdout_records=len(test),
        optimizer_steps=a.steps,gradient_at_start=gradient,measurements=measurements,production_model_changed=False,
        frozen_encoder=True,teacher_inputs=False,scope='Frozen-input overfit/readability diagnostic, not native accuracy',
        training_physical=sorted({r['physical'] for r in train}),holdout_physical=sorted({r['physical'] for r in test})),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cache','fit']);p.add_argument('--out',required=True)
    p.add_argument('--config',default='configs/jepa/geometry_transport_conditioned_v36.yaml');p.add_argument('--rank',type=int,default=0)
    p.add_argument('--cache');p.add_argument('--count',type=int,default=8);p.add_argument('--loss',choices=['flow','mixed'],default='flow');p.add_argument('--steps',type=int,default=400)
    a=p.parse_args();torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    (cache if a.mode=='cache' else fit)(a)


if __name__=='__main__':main()
