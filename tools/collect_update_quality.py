"""Frozen LIP proposal outcomes; physical train/calibration/audit separation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.data.stream_clips import StreamClips,collate
from lip.engine.config import check_data_gate
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import load_init,sha,source_hash
from lip.engine.stream_training import StreamTrainingModule
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import update,angle
from lip.models.update_quality import observation_vector


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False);c=load_stream_config(a.config);audit=check_data_gate(a.index_root)
    ds=StreamClips(a.data_root,a.index_root,c['burn_in_frames'],c['supervised_unroll_frames']);rng=np.random.default_rng(20260914)
    groups={p:{} for p in ('train','calibration','audit')};physical_sets={p:set() for p in groups}
    for obj,physical in ds.groups.items():
        names=sorted(physical);rng.shuffle(names);n=max(1,len(names)//10)
        if len(names)<3:raise ValueError('Insufficient physical sequences for three-way split')
        for part,names_part in [('audit',names[:n]),('calibration',names[n:2*n]),('train',names[2*n:])]:
            groups[part][obj]={name:physical[name] for name in names_part};physical_sets[part].update(names_part)
    assert not any(physical_sets[x]&physical_sets[y] for x,y in [('train','calibration'),('train','audit'),('audit','calibration')])
    manifest=[]
    for part,count in [('train',1536),('calibration',256),('audit',256)]:
        for _ in range(count):
            obj=int(rng.choice(sorted(groups[part])));physical=groups[part][obj];sid=sorted(physical)[int(rng.integers(len(physical)))];stream=int(rng.choice(physical[sid]))
            manifest.append(dict(stream=stream,start=int(rng.choice(ds.starts[stream])),seed=int(rng.integers(2**31)),partition=part,physical_sequence=sid,stream_id=ds.streams[stream]['stream_id']))
    (a.out/'samples.json').write_text(json.dumps(manifest))
    report=dict(phase='prepared',teacher_checkpoint=str(a.checkpoint.resolve()),teacher_sha256=sha(a.checkpoint),config=str(a.config.resolve()),
        source_sha256=source_hash(),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],data_root=str(a.data_root.resolve()),index_root=str(a.index_root.resolve()),
        sampling_manifest_sha256=sha(a.out/'samples.json'),physical_sequences={p:sorted(v) for p,v in physical_sets.items()},
        split='official train only; three disjoint physical-sequence partitions for the quality head',clips={p:sum(m['partition']==p for m in manifest) for p in groups},
        frames_per_clip=c['supervised_unroll_frames'],candidate_scales=[-1.,0.,.5,1.,1.5],original_candidate=3,hold_candidate=1,
        error='ADD-S / diameter on the fixed 512 sampled mesh surface points; all continuous candidate errors retained',improvement_margin=.005,
        gt_used_for_labels_only=True,foundationpose_calls=0,hand_supervision=False)
    (a.out/'collection.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='physical_sequences'},indent=2))


def collect(a):
    report=json.loads((a.out/'collection.json').read_text());assert source_hash()==report['source_sha256'] and sha(a.checkpoint)==report['teacher_sha256']
    manifest=json.loads((a.out/'samples.json').read_text());assert sha(a.out/'samples.json')==report['sampling_manifest_sha256']
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(42)
    c=load_stream_config(a.config);model=make_model(c).cuda().eval();load_init(a.checkpoint,model,check_data_gate(a.index_root),c)
    positions=list(range(a.rank,len(manifest),a.world));ds=StreamClips(a.data_root,a.index_root,c['burn_in_frames'],c['supervised_unroll_frames'],
        fixed=manifest,rank=a.rank,world=a.world,length=len(positions),decode_threads=4)
    loader=DataLoader(ds,batch_size=16,collate_fn=collate,num_workers=2,prefetch_factor=2,pin_memory=True)
    renderer=Renderer('cuda');runner=StreamTrainingModule(model,c,renderer);captured=[];frame_counter=0
    def hook(module,args,result):
        nonlocal frame_counter
        features=args[0];out=result[0]
        if frame_counter>=c['burn_in_frames']:
            observation,delta=observation_vector(features,out)
            captured.append((observation.detach(),delta.detach(),features['T_base_centered'].detach(),out['pose_centered'].detach(),frame_counter))
        frame_counter+=1
    handle=model.register_forward_hook(hook);dest=a.out/f'rank{a.rank}';dest.mkdir(exist_ok=False);records=0
    scales=torch.tensor(report['candidate_scales'],device='cuda')
    with torch.no_grad(),(dest/'progress.jsonl').open('w') as log:
        for batch_index,samples in enumerate(loader):
            started=time.time();captured=[];frame_counter=0;runner(samples)
            diameter=torch.tensor([float(s['mesh']['diameter']) for s in samples],device='cuda');points=torch.stack([torch.as_tensor(s['mesh']['points'],device='cuda') for s in samples]);b=len(samples)
            chunks=[]
            for observation,delta,base,original_pose,i in captured:
                # Candidate GT labels are formed after the actor trajectory is
                # complete. The wrapper's pose loss never changes actor state.
                gt=torch.stack([s['targets'][i].cuda() for s in samples]);gtpoints=points@gt[:,:3,:3].transpose(-1,-2)+gt[:,None,:3,3]
                candidates=[];adds=[];add=[];rot=[];ctr=[]
                actions=delta[:,None]*scales[None,:,None]
                for j in range(len(scales)):
                    pose=update(base.float(),actions[:,j,:3].float(),actions[:,j,3:].float(),diameter)
                    if j==report['original_candidate']:torch.testing.assert_close(pose,original_pose,rtol=0,atol=1e-6)
                    pts=points@pose[:,:3,:3].transpose(-1,-2)+pose[:,None,:3,3]
                    adds.append(torch.cdist(pts,gtpoints,compute_mode='donot_use_mm_for_euclid_dist').min(-1).values.mean(-1)/diameter)
                    add.append((pts-gtpoints).norm(dim=-1).mean(-1)/diameter)
                    rot.append(angle(pose[:,:3,:3]@gt[:,:3,:3].transpose(-1,-2))*180/torch.pi);ctr.append((pose[:,:3,3]-gt[:,:3,3]).norm(dim=-1)*1000);candidates.append(pose)
                errors=torch.stack(adds,1);before=errors[:,report['hold_candidate']]
                chunk=dict(observation=observation.half().cpu(),delta=actions.cpu(),candidate_pose=torch.stack(candidates,1).cpu(),base_pose=base.cpu(),gt_pose=gt.cpu(),
                    adds_d=errors.cpu(),add_d=torch.stack(add,1).cpu(),rotation_deg=torch.stack(rot,1).cpu(),center_mm=torch.stack(ctr,1).cpu(),diameter=diameter.cpu(),
                    advantage=(before[:,None]-errors).cpu(),harmful=(errors>before[:,None]+.005).cpu(),improve=(errors<before[:,None]-.005).cpu(),converge=(errors<.1).cpu(),
                    partition=[s['sample']['partition'] for s in samples],physical_sequence=[s['sample']['physical_sequence'] for s in samples],
                    stream_id=[s['stream']['stream_id'] for s in samples],frame=[int(s['frames'][i+1]) for s in samples])
                chunks.append(chunk)
            out={key:torch.cat([v[key] for v in chunks]) if isinstance(chunks[0][key],torch.Tensor) else sum([v[key] for v in chunks],[]) for key in chunks[0]}
            path=dest/f'batch_{batch_index:04d}.pt';torch.save(out,path);records+=len(out['observation'])
            log.write(json.dumps(dict(batch=batch_index,observations=records,seconds=time.time()-started,file=path.name,sha256=sha(path)))+'\n');log.flush()
    handle.remove();(dest/'completed.json').write_text(json.dumps(dict(completed=True,observations=records,rank=a.rank,teacher_sha256=report['teacher_sha256'],source_sha256=source_hash()),indent=2))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('config','checkpoint','data-root','index-root','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8);a=p.parse_args()
    prepare(a) if a.prepare else collect(a)


if __name__=='__main__':main()
