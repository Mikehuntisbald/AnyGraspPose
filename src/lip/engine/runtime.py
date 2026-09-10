import contextlib
import time
import torch
from lip.data.perturb import noisy_history
from lip.engine.features import build_features, stack_features
from lip.losses import pose_loss


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def batch_step(model, batch, renderer, config, rollout=1, noise=True, backward=True,
               loss_scale=1., sync_final=True):
    device=next(model.parameters()).device;length=config['clip_length']
    times=dict(render_time=0.,forward_time=0.,backward_time=0.)
    states=[]
    for item in batch:
        g=torch.Generator().manual_seed(item['seed'])
        # Only strict past GT seeds the first update. Later slots are model-owned.
        initial,clipped=noisy_history(item['poses'][:length].to(device),float(item['mesh']['diameter']),g,noise)
        expansion=float(config.get('crop_expansion',2.)*(.8+.4*torch.rand((),generator=g))) if noise and config.get('augmentation',True) else config.get('crop_expansion',2.)
        states.append(dict(accepted=list(initial.unbind()),generator=g,clipped=clipped,expansion=expansion))
    metrics=[];outputs=[]
    for u in range(rollout):
        sync(device);t=time.perf_counter();features=[];targets=[];points=[]
        for item,state in zip(batch,states):
            base=state['accepted'][-1]
            past=state['accepted'][u+1:u+length]
            hist=torch.stack(past+[torch.eye(4,device=device)])
            valid=torch.arange(length,device=device)>=max(0,length-item['effective']-u)
            pv=valid.clone();pv[-1]=False
            if noise:
                dropout=torch.rand(length,generator=state['generator']).to(device)<config.get('pose_token_dropout',.2)
                pv=pv & ~dropout
            f,_=build_features(item['rgb'][u:u+length].to(device),item['depth'][u:u+length].to(device),hist,
                               item['times'][u:u+length].to(device),valid,pv,base,item['k'].to(device),item['mesh'],renderer,
                               config['image_size'],state['expansion'])
            features.append(f);targets.append(item['poses'][length+u].to(device));points.append(torch.as_tensor(item['mesh']['points'],device=device))
        inp=stack_features(features);sync(device);times['render_time']+=time.perf_counter()-t
        final=(u==rollout-1 and sync_final)
        context=model.no_sync() if hasattr(model,'no_sync') and not final else contextlib.nullcontext()
        with context:
            t=time.perf_counter()
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                out=model(**inp)
            loss,ls=pose_loss(out['pose_centered'],torch.stack(targets),torch.stack(points),inp['object_diameter_m'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pose loss')
            sync(device);times['forward_time']+=time.perf_counter()-t
            if backward:
                t=time.perf_counter();(loss*loss_scale/rollout).backward();sync(device);times['backward_time']+=time.perf_counter()-t
        metrics.append(dict(loss=float(loss.detach()),**{k:float(v.detach()) for k,v in ls.items()},
                            update_rotation_rad=float(out['delta_rotvec'].detach().norm(dim=-1).mean()),
                            update_center_norm=float(out['delta_center_norm'].detach().norm(dim=-1).mean())))
        for state,pred in zip(states,out['pose_centered'].detach()):state['accepted'].append(pred)
        outputs.append(out['pose_centered'].detach())
    values={k:sum(m[k] for m in metrics)/rollout for k in metrics[0]}
    values.update(times)
    values.update({k:sum(s['clipped'][k] for s in states)/len(states) for k in states[0]['clipped']})
    return values,outputs
