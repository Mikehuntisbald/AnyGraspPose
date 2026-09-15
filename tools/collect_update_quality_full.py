"""Collect complete frozen-actor streams, including drift and rejected steps."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.data.index import read_frame
from lip.data.perturb import noisy_history
from lip.engine.config import check_data_gate
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import load_init,sha,source_hash
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import update,angle,center_pose,original_pose
from lip.models.update_quality import observation_vector


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False)
    old=json.loads((a.reference_cache/'collection.json').read_text());manifest=json.loads((a.reference_cache/'samples.json').read_text())
    assert sha(a.checkpoint)==old['teacher_sha256']
    audit=check_data_gate(a.index_root);assert all(audit[k]==old[k] for k in ('split_hash','mesh_hash'))
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    for i,item in enumerate(manifest):
        s=streams[item['stream_id']];assert s['split']=='train'
        assert item['physical_sequence'] in old['physical_sequences'][item['partition']]
        item.update(start=0,manifest_index=i,expected_frames=s['num_frames']-1)
    (a.out/'samples.json').write_text(json.dumps(manifest))
    report=dict(old,phase='prepared',source_sha256=source_hash(),collector_sha256=sha(__file__),world_size=a.world,
        reference_cache=str(a.reference_cache.resolve()),reference_manifest_sha256=sha(a.reference_cache/'samples.json'),
        sampling_manifest_sha256=sha(a.out/'samples.json'),expected_observations=sum(x['expected_frames'] for x in manifest),
        frames_per_clip=None,trajectory='Full stream from its first frame; own accepted predictions, no burn-in exclusion, no GT resets',
        physical_partition_reused=True,audit_previously_inspected=True,validation_scope='Development audit on the same physically disjoint head partitions; not a fresh confirmatory test')
    (a.out/'collection.json').write_text(json.dumps(report,indent=2));print('Prepared',report['expected_observations'],'frame opportunities',flush=True)


@torch.no_grad()
def label(observation,delta,base,teacher_pose,gt,mesh,scales):
    points=mesh['points'][None].float();diameter=mesh['diameter'].float().reshape(1)
    gt=gt[None].float();ground=points@gt[:,:3,:3].transpose(-1,-2)+gt[:,None,:3,3]
    actions=delta[:,None]*scales[None,:,None];poses=[];adds=[];add=[];rot=[];ctr=[]
    for j in range(len(scales)):
        pose=update(base.float(),actions[:,j,:3].float(),actions[:,j,3:].float(),diameter)
        if j==3:torch.testing.assert_close(pose,teacher_pose,rtol=0,atol=1e-6)
        xyz=points@pose[:,:3,:3].transpose(-1,-2)+pose[:,None,:3,3]
        adds.append(torch.cdist(xyz,ground,compute_mode='donot_use_mm_for_euclid_dist').min(-1).values.mean(-1)/diameter)
        add.append((xyz-ground).norm(dim=-1).mean(-1)/diameter)
        rot.append(angle(pose[:,:3,:3]@gt[:,:3,:3].transpose(-1,-2))*180/torch.pi)
        ctr.append((pose[:,:3,3]-gt[:,:3,3]).norm(dim=-1)*1000);poses.append(pose)
    errors=torch.stack(adds,1);before=errors[:,1]
    return dict(observation=observation.half().cpu(),delta=actions.cpu(),candidate_pose=torch.stack(poses,1).cpu(),base_pose=base.cpu(),gt_pose=gt.cpu(),
        adds_d=errors.cpu(),add_d=torch.stack(add,1).cpu(),rotation_deg=torch.stack(rot,1).cpu(),center_mm=torch.stack(ctr,1).cpu(),diameter=diameter.cpu(),
        advantage=(before[:,None]-errors).cpu(),harmful=(errors>before[:,None]+.005).cpu(),improve=(errors<before[:,None]-.005).cpu(),converge=(errors<.1).cpu())


def collect(a):
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(42)
    import cv2
    cv2.setNumThreads(1)
    report=json.loads((a.out/'collection.json').read_text());assert report['source_sha256']==source_hash() and report['collector_sha256']==sha(__file__)
    assert sha(a.checkpoint)==report['teacher_sha256'] and sha(a.out/'samples.json')==report['sampling_manifest_sha256']
    c=load_stream_config(a.config);audit=check_data_gate(a.index_root);model=make_model(c).cuda().eval();load_init(a.checkpoint,model,audit,c)
    renderer=Renderer('cuda');streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    manifest=json.loads((a.out/'samples.json').read_text())[a.rank::a.world]
    if a.limit_streams:manifest=manifest[:a.limit_streams]
    dest=a.out/f'rank{a.rank}';dest.mkdir(exist_ok=False)
    captured=[]
    def hook(module,args,result):
        features=args[0];out=result[0];obs,delta=observation_vector(features,out)
        captured.append((obs.detach(),delta.detach(),features['T_base_centered'].detach(),out['pose_centered'].detach()))
    handle=model.register_forward_hook(hook);scales=torch.tensor(report['candidate_scales'],device='cuda');records=failed=opportunities=rejected=0
    with torch.no_grad(),ThreadPoolExecutor(2) as decoder,(dest/'progress.jsonl').open('w') as log,(dest/'failures.jsonl').open('w') as failures:
        for batch_index,item in enumerate(manifest):
            started=time.time();s=streams[item['stream_id']]
            with np.load(a.index_root/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            with np.load(a.index_root/s['pose_cache']) as z:poses=z['poses'].copy();frames=z['frames'].copy();times=z['timestamps'].copy() if 'timestamps' in z else frames.astype('f8')/audit['fps']
            init=center_pose(torch.from_numpy(poses[0]),torch.from_numpy(mesh['center']))
            init,_=noisy_history(init[None],float(mesh['diameter']),torch.Generator().manual_seed(item['seed']))
            state=model.initialize(original_pose(init[0],torch.from_numpy(mesh['center'])),mesh,s['intrinsics'],s['stream_id'],times[0],object_id=s['object_id'],camera_id=s['camera_serial'])
            # Only image decoding is prefetched. Future pixels never enter model.step.
            images=decoder.map(lambda frame:read_frame(a.data_root,s,int(frame),audit['depth_scale_to_m']),frames[1:])
            chunks=[];stream_failed=stream_rejected=0
            for j,(rgb,depth) in enumerate(images,1):
                captured.clear();proposal,candidate=model.step(torch.from_numpy(rgb),torch.from_numpy(depth),times[j],state,renderer=renderer,precision=c['precision'],image_size=c['image_size'],crop_expansion=c['crop_expansion'])
                if proposal['status']=='ok':state=model.commit(proposal,candidate)
                else:stream_rejected+=1
                opportunities+=1
                usable=bool(captured) and bool(torch.stack([torch.isfinite(t).all() for t in captured[0]]).all())
                if not usable:
                    failures.write(json.dumps(dict(manifest_index=item['manifest_index'],stream_id=s['stream_id'],frame=int(frames[j]),status=proposal['status'],reason='No finite actor proposal available',partition=item['partition']))+'\n');failures.flush();stream_failed+=1;continue
                gt=center_pose(torch.from_numpy(poses[j]).cuda(),state.mesh['center'])
                chunk=label(*captured[0],gt,state.mesh,scales)
                chunk.update(partition=[item['partition']],physical_sequence=[item['physical_sequence']],stream_id=[s['stream_id']],frame=[int(frames[j])],
                    manifest_index=[item['manifest_index']],proposal_status=[proposal['status']],stream_position=[j])
                chunks.append(chunk)
            assert len(chunks)+stream_failed==item['expected_frames'];failed+=stream_failed;rejected+=stream_rejected
            out={key:torch.cat([v[key] for v in chunks]) if isinstance(chunks[0][key],torch.Tensor) else sum([v[key] for v in chunks],[]) for key in chunks[0]} if chunks else None
            row=dict(batch=batch_index,manifest_index=item['manifest_index'],stream_id=s['stream_id'],frames=item['expected_frames'],unusable_frames=stream_failed,rejected_frames=stream_rejected,seconds=time.time()-started)
            if out is not None:
                path=dest/f'batch_{batch_index:04d}.pt';torch.save(out,path);records+=len(out['observation']);row.update(file=path.name,sha256=sha(path))
            row['observations']=records;log.write(json.dumps(row)+'\n');log.flush()
    handle.remove();assert records+failed==opportunities
    (dest/'completed.json').write_text(json.dumps(dict(completed=True,smoke=a.limit_streams is not None,observations=records,unusable_frames=failed,rejected_frames=rejected,frame_opportunities=opportunities,rank=a.rank,teacher_sha256=report['teacher_sha256'],source_sha256=source_hash()),indent=2))


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','checkpoint','data-root','index-root','out','reference-cache'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=32);p.add_argument('--limit-streams',type=int);a=p.parse_args()
    prepare(a) if a.prepare else collect(a)


if __name__=='__main__':main()
