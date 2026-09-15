"""Read-only local controllability of the frozen pose head on cached train states.

GT-directed latent inversion is an oracle capacity diagnostic, never inference
or a tracking result. The saved actor histories and all model weights stay fixed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F


def value_jacobian(weights, z):
    w1,b1,w2,b2=weights
    x=F.linear(z,w1,b1)
    derivative=.5*(1+torch.erf(x/math.sqrt(2)))+x*torch.exp(-.5*x.square())/math.sqrt(2*math.pi)
    return F.linear(F.gelu(x),w2,b2),torch.einsum('oh,bh,hi->boi',w2,derivative,w1)


@torch.no_grad()
def invert_head(weights, z, gain, target, steps=12, radius=1.):
    start=z.clone();z=z.clone();initial=None
    for _ in range(steps):
        value,j=value_jacobian(weights,z);value=value*gain[:,None];j=j*gain[:,None,None]
        error=target-value
        if initial is None:initial=error.norm(dim=-1)
        gram=j@j.transpose(-1,-2)+1e-10*torch.eye(6,dtype=j.dtype,device=j.device)
        shift=(j.transpose(-1,-2)@torch.linalg.solve(gram,error[...,None])).squeeze(-1)
        shift=shift*(radius/shift.norm(dim=-1).clamp_min(radius))[:,None]
        candidates=torch.stack([z+scale*shift for scale in (1.,.5,.25,.125,0.)],1)
        w1,b1,w2,b2=weights
        values=F.linear(F.gelu(F.linear(candidates,w1,b1)),w2,b2)*gain[:,None,None]
        residual=(values-target[:,None]).square().sum(-1)
        choose=residual.argmin(1)
        z=candidates[torch.arange(len(z)),choose]
    value,_=value_jacobian(weights,z)
    return dict(latent=z,value=value*gain[:,None],shift_norm=(z-start).norm(dim=-1),initial_error=initial,
        final_error=(value*gain[:,None]-target).norm(dim=-1))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    x=np.asarray(values,dtype='f8')
    return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max())) if len(x) else None


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('runtime','cache','reference-checkpoint','visibility','index-root','out'):
        p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.runtime/'src'))
    from lip.geometry.so3 import log as so3_log
    torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False)
    collection=json.loads((a.cache/'collection.json').read_text());assert sha(collection['teacher_checkpoint'])==collection['teacher_sha256']
    sampling=json.loads((a.cache/'samples.json').read_text());assert sha(a.cache/'samples.json')==collection['sampling_manifest_sha256']
    teacher=torch.load(collection['teacher_checkpoint'],map_location='cpu',weights_only=False)
    reference=torch.load(a.reference_checkpoint,map_location='cpu',weights_only=False)
    keys=['head.0.weight','head.0.bias','head.2.weight','head.2.bias']
    assert all(torch.equal(teacher['model'][key],reference['model'][key]) for key in keys)
    weights=[teacher['model'][key].double().clone() for key in keys]
    update_strength=float(teacher['model']['update_strength']);del teacher,reference
    vm=json.loads((a.visibility/'completed.json').read_text());assert vm['completed'] and vm['split']=='train' and vm['output_sha256']==sha(a.visibility/'train_visibility.jsonl')
    assert all(collection[k]==vm[k] for k in ('split_hash','mesh_hash'))
    visibility={row['stream_id']:dict(zip(row['frames'],row['values'])) for row in map(json.loads,(a.visibility/'train_visibility.jsonl').read_text().splitlines())}
    index={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    allowed={i['manifest_index'] for i in sampling if i['partition']=='audit'}
    parts={k:[] for k in ('z','gain','base','gt','diameter','action','adds')};metadata=[];files=[]
    for rank in range(collection['world_size']):
        folder=a.cache/f'rank{rank}';done=json.loads((folder/'completed.json').read_text());assert done['completed'] and done['teacher_sha256']==collection['teacher_sha256']
        for row in map(json.loads,(folder/'progress.jsonl').read_text().splitlines()):
            if row['manifest_index'] not in allowed or 'file' not in row:continue
            path=folder/row['file'];assert sha(path)==row['sha256'];data=torch.load(path,map_location='cpu',weights_only=False)
            assert all(part=='audit' for part in data['partition']) and data['observation'].shape[1]==815
            obs=data['observation'].double();parts['z'].append(obs[:,:256]);parts['gain'].append(1-.5*math.tanh(update_strength)*(1-obs[:,808]))
            for field,key in [('base','base_pose'),('gt','gt_pose'),('diameter','diameter')]:parts[field].append(data[key].double())
            parts['action'].append(data['delta'][:,3].double());parts['adds'].append(data['adds_d'][:,3].double())
            for sid,frame,position in zip(data['stream_id'],data['frame'],data['stream_position']):
                assert index[sid]['split']=='train'
                metadata.append(dict(stream_id=sid,frame=frame,stream_position=position,object_id=index[sid]['object_id'],visibility=visibility[sid][frame]))
            files.append(dict(path=str(path),sha256=row['sha256']))
    data={key:torch.cat(values) for key,values in parts.items()};n=len(metadata);assert all(len(t)==n for t in data.values())
    # Fixed bounded population: every severe cached state, 16 largest cached
    # actor errors per object, and 512 uniform states. Overlap is removed.
    rng=np.random.default_rng(20260914);selected=set(rng.choice(n,size=min(n,512),replace=False).tolist())
    selected.update(i for i,r in enumerate(metadata) if r['visibility'] is not None and r['visibility']<.3)
    errors=data['adds'].numpy()
    for obj in sorted({r['object_id'] for r in metadata}):
        ids=[i for i,r in enumerate(metadata) if r['object_id']==obj]
        selected.update(sorted(ids,key=lambda i:(-errors[i],i))[:16])
    selected=sorted(selected);records=[]
    for offset in range(0,len(selected),128):
        ids=selected[offset:offset+128];z=data['z'][ids];gain=data['gain'][ids];base=data['base'][ids];gt=data['gt'][ids]
        target=torch.cat((so3_log(gt[:,:3,:3]@base[:,:3,:3].transpose(-1,-2)),(gt[:,:3,3]-base[:,:3,3])/data['diameter'][ids,None]),-1)
        value,j=value_jacobian(weights,z);value=value*gain[:,None];j=j*gain[:,None,None]
        singular=torch.linalg.svdvals(j)
        error=target-value
        local=(j.transpose(-1,-2)@torch.linalg.solve(j@j.transpose(-1,-2)+1e-10*torch.eye(6,dtype=j.dtype),error[...,None])).squeeze(-1)
        inverse=invert_head(weights,z,gain,target)
        for lane,idx in enumerate(ids):
            r=dict(metadata[idx],cache_observation_index=idx,cached_actor_adds_d=float(data['adds'][idx]),latent_norm=float(z[lane].norm()),
                minimum_singular_value=float(singular[lane,-1]),maximum_singular_value=float(singular[lane,0]),
                condition_number=float(singular[lane,0]/singular[lane,-1]),rank=int((singular[lane]>singular[lane,0]*1e-6).sum()),
                cpu_float64_vs_cached_action_l2=float((value[lane]-data['action'][idx]).norm()),
                action_residual_before=float(inverse['initial_error'][lane]),action_residual_after=float(inverse['final_error'][lane]),
                linearized_latent_shift_norm=float(local[lane].norm()),oracle_latent_shift_norm=float(inverse['shift_norm'][lane]))
            records.append(r)
    summary={}
    for name,rows in [('all_selected',records),('natural_visibility_lt03',[r for r in records if r['visibility'] is not None and r['visibility']<.3]),
                      ('cached_actor_adds_ge005',[r for r in records if r['cached_actor_adds_d']>=.05])]:
        summary[name]=dict(frames=len(rows),rank6_frames=sum(r['rank']==6 for r in rows),oracle_action_residual_below_1e_minus5=sum(r['action_residual_after']<1e-5 for r in rows),
            **{k:stats([r[k] for r in rows]) for k in ('minimum_singular_value','condition_number','latent_norm','linearized_latent_shift_norm','oracle_latent_shift_norm','action_residual_before','action_residual_after','cpu_float64_vs_cached_action_l2')})
    (a.out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    report=dict(completed=True,scope='Read-only capacity diagnostic on a previously inspected development-audit partition INSIDE official train. Cached R1K1 histories; not current-parent trajectories, independent validation, learned prediction, closed-loop performance, or an inference algorithm.',
        teacher_sha256=collection['teacher_sha256'],reference_checkpoint_sha256=sha(a.reference_checkpoint),head_tensors_bitwise_equal=True,
        cache_collection_sha256=sha(a.cache/'collection.json'),visibility_sha256=vm['output_sha256'],tool_sha256=sha(__file__),
        selection='Every cached natural visibility <.3 frame, 16 largest actor ADD-S errors per object, and 512 uniform states; fixed seed 20260914; union without duplicates',
        cached_observations=n,selected_observations=len(records),selected_objects=sorted({r['object_id'] for r in records}),
        precision_limit='Observations were stored in float16 after the original BF16 actor. Jacobians and inversion use CPU float64; recomputed action differences are explicitly reported.',
        oracle='12 iterations, latent trust radius 1 per iteration, damped minimum-norm Newton direction and non-increasing action-space line search. GT enters this diagnostic only. No weights changed.',
        action_units='Rotation vector radians and center displacement / object diameter; Euclidean six-dimensional action residual is a diagnostic, not the pose training loss.',
        summaries=summary,cache_files=files,frames_sha256=sha(a.out/'frames.jsonl'))
    (a.out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in report.items() if k!='cache_files'},indent=2))


if __name__=='__main__':main()
