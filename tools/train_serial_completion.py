"""Bounded V21 training: paired estimates, exact-zero examples, student feedback.

Independent RNG per rank, one packed gradient collective, complete resumable
checkpoints. Oracle completion is never used as an input in this stage.
"""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np,torch,torch.distributed as dist,yaml
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_serial_completion import initialize,heldout
from lip.unified.build import make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.unified.cad_surface_targets import surface_targets
from lip.unified.serial_objective import objective,delta_target
from lip.unified.reconstruction_only import is_pose_parameter
from lip.unified.optimized_training import synchronize_gradients
from lip.unified.ema_encoder import update_ema
from lip.unified.checkpoint import save,resume,atomic_json
from lip.engine.jepa_checkpoint import sha
from lip.engine.config import check_data_gate
from lip.geometry.so3 import update


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--oracle',required=True)
    p.add_argument('--stop-at',type=int);p.add_argument('--resume');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    c=yaml.safe_load(Path(a.config).read_text());rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if a.preflight:c['runtime']['formal']=False
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(c['seed']);random.seed(c['seed']+rank);np.random.seed(c['seed']+rank)
    torch.use_deterministic_algorithms(c['runtime'].get('deterministic_algorithms',True))
    torch.utils.deterministic.fill_uninitialized_memory=False
    if world>1:dist.init_process_group('nccl')
    if not a.preflight and (world!=c['runtime']['world'] or world*c['runtime']['microbatch']!=c['training']['effective_batch']):
        raise ValueError('Formal effective batch/world mismatch')
    gate=json.loads((Path(a.oracle)/'receipt.json').read_text())
    if not gate['passed'] or sha(Path(a.oracle)/'readout.pt')!=gate['checkpoint_sha256']:raise ValueError('Oracle readout gate not passed')
    model,source,new=initialize(c);del source
    oracle=torch.load(Path(a.oracle)/'readout.pt',map_location='cpu',weights_only=False)
    states=model.state_dict()
    for key,value in oracle['model'].items():
        if not is_pose_parameter(key) or key not in states or states[key].shape!=value.shape:raise ValueError('Readout migration mismatch: '+key)
        states[key].copy_(value)
    del states,oracle
    for name,parameter in model.named_parameters():
        active=not (name.startswith(('ema_teacher.','writer.','memory_position.','core.memory_')) or '.history.' in name or name.startswith('core.log_error.'))
        parameter.requires_grad_(active)
    groups={}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:continue
        kind='pose' if is_pose_parameter(name) else ('encoder' if name.startswith('encoder.') else ('new' if name.startswith(('surface_head.','cad_surface.','core.feature_')) else 'predictor'))
        groups.setdefault(kind,dict(params=[],names=[],lr=c['training']['learning_rates'][kind],category=kind))
        groups[kind]['params'].append(parameter);groups[kind]['names'].append(name)
    opt=torch.optim.AdamW(list(groups.values()),weight_decay=.01,fused=True)
    maximum=c['serial_completion']['joint_steps'];stop=a.stop_at or maximum
    if not 0<stop<=maximum:raise ValueError('Stop exceeds bounded budget')
    def rate(step):return min(1.,(step+1)/50)*(.3+.7*.5*(1+math.cos(math.pi*min(step,maximum)/maximum)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,rate)
    factory=Factory(c,model,make_store(c,model));audit=check_data_gate(c['paths']['index_root'])
    source_files={str(f.relative_to(Path(__file__).resolve().parents[1])):sha(f) for d in ('src','tools','configs') for f in (Path(__file__).resolve().parents[1]/d).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    provenance=dict(split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],source_sha256=hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest(),
        initializers_sha256=factory.initializers_sha256,weights=c['weights'],source_checkpoint_sha256=c['serial_completion']['source_sha256'],
        oracle_readout_sha256=gate['checkpoint_sha256'],training_split='train',official_test_access=False,teacher_input=False,
        crop_reference='student own detached feedback; GT noisy/zero pose curricula are training-only',paired_estimates=True)
    out=Path(c['paths']['output']);out=out.parent.parent/'preflight_run' if a.preflight else out
    if rank==0:out.mkdir(parents=True,exist_ok=bool(a.resume));atomic_json(out/'provenance.json',provenance)
    if world>1:dist.barrier()
    start=0
    if a.resume:start=resume(a.resume,model,opt,scheduler,c,provenance,rank,world)['step']
    else:save(out/'initial.pt',model,opt,scheduler,0,c,provenance)
    batch=1 if a.preflight else c['runtime']['microbatch']
    # Keep rank model initialization equal; rank RNG differs only afterwards.
    if not a.resume:torch.manual_seed(c['seed']+rank)
    def episodes(step):
        result=[]
        for lane in range(batch):
            seed=9000000+(step*world+rank)*batch+lane
            for attempt in range(100):
                sample=factory.sample(seed+attempt*100000003,frames=12)
                if not heldout(sample[0].stream):result.append(sample);break
            else:raise RuntimeError('Cannot draw training sequence outside gate holdout')
        return result
    def forward(episodes,targets,bases,frame):
        scenes=[prepare_scene(e.rgb[frame],e.depth[frame],bases[i],e.mesh,e.k,e.times[frame],e.stream+f'/lane{i}',e.cad,factory.renderer,fast=True) for i,e in enumerate(episodes)]
        occ=[e.occlusion_plan.render(s,frame) for e,s in zip(episodes,scenes)]
        obs=encode_scenes(model,scenes,occlusions=occ,frame_id=frame)
        truth=torch.stack([t[0][frame] for t in targets]);mask=torch.stack([t[1][frame] for t in targets])
        teacher=build_teachers(model.ema_teacher,scenes,truth,mask,[o.mask for o in occ],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True)
        teacher=surface_targets(model,scenes,teacher)
        with torch.autocast('cuda',dtype=torch.bfloat16):output,_=model(obs)
        points=torch.stack([torch.as_tensor(e.mesh['points'],device='cuda') for e in episodes])
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(output,teacher,truth,points,obs.diameter,bases,obs.measured_depth_m,c['training']['loss_weights'])
        return output,loss,metrics,obs,teacher
    path_norms=None
    with (out/f'rank{rank}.jsonl').open('a') as log:
        for step in range(start,stop):
            begun=time.monotonic();pairs=episodes(step);ep,targets=zip(*pairs)
            frame=8 if step%2==0 else 0
            truth=torch.stack([t[0][frame] for t in targets]);diam=truth.new_tensor([float(e.mesh['diameter']) for e in ep])
            noise=torch.randn(batch,6,device='cuda')*truth.new_tensor([.15,.15,.15,.035,.035,.035])
            bases=update(truth,noise[:,:3],noise[:,3:],diam)
            # Native, potentially large errors remain represented; never run an
            # immutable reference trajectory or inspect GT during deployment.
            if step%4 in (2,3):bases=torch.stack([e.initial for e in ep])
            opt.zero_grad(set_to_none=True)
            first,l1,m1,obs,target=forward(ep,targets,bases,frame)
            if a.preflight:
                from lip.losses import pose_loss
                pure_pose,_=pose_loss(first['pose_centered'],truth,torch.stack([torch.as_tensor(e.mesh['points'],device='cuda') for e in ep]),diam)
                gradients=torch.autograd.grad(pure_pose,[first['surface_xyz'],first['surface_depth_residual'],first['f_mid_predicted'],first['f_predicted'],first['patch_latent']],retain_graph=True)
                path_norms=[float(x.norm()) for x in gradients]
                if not all(x>0 and math.isfinite(x) for x in path_norms):raise AssertionError('Pose did not consume every completion output')
            feedback=first['pose_centered'].detach()
            # Same RGB-D observation, alternate estimated pose. Every fourth
            # example is an exact-zero base; its target is always a zero update.
            alternative=truth if step%4==0 else update(bases,-noise[:,:3],-noise[:,3:],diam)
            second,l2,m2,_,_=forward(ep,targets,alternative,frame)
            pred1=torch.cat((first['delta_rotvec'],first['delta_center_norm']),-1)
            pred2=torch.cat((second['delta_rotvec'],second['delta_center_norm']),-1)
            expected=delta_target(bases,truth,diam)-delta_target(alternative,truth,diam)
            scale=truth.new_tensor([.174533]*3+[.05]*3)
            pair=F.smooth_l1_loss((pred1-pred2)/scale,expected/scale,beta=.1)
            combined=(l1+l2)/3+.1*pair
            combined.backward();del first,second,obs,target,l1,l2,combined,pred1,pred2
            third,l3,m3,_,_=forward(ep,targets,feedback,frame+1)
            (l3/3).backward();loss_value=float(l3.detach());del third,l3
            communication=time.monotonic();synchronize_gradients(model.parameters());communication=time.monotonic()-communication
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None],1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient; do not advance')
            opt.step();scheduler.step();update_ema(model)
            torch.cuda.synchronize();metrics={k:float((m1[k]+m2[k]+m3[k])/3) for k in m1}
            row=dict(step=step+1,source_step=45400,seconds=time.monotonic()-begun,loss_feedback=loss_value,pair=float(pair.detach()),
                metrics=metrics,gradient_norm=float(norm),communication_seconds=communication,
                learning_rates={g['category']:g['lr'] for g in opt.param_groups},peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if rank==0 and (step<3 or (step+1)%25==0):print(json.dumps(row),flush=True)
            if (step+1)%c['serial_completion']['checkpoint_every']==0 or step+1==stop:save(out/'last.pt',model,opt,scheduler,step+1,c,provenance)
    if rank==0:atomic_json(out/'completion.json',dict(completed=True,step=stop,preflight=a.preflight,world=world,gradient_path_norms=path_norms,checkpoint_sha256=sha(out/'last.pt'),official_test_access=False))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
