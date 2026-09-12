import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from lip.data.index import read_frame
from lip.data.clips import ClipDataset
from lip.data.perturb import noisy_history
from lip.data.motion import motion_thresholds
from lip.engine.config import load_config,check_data_gate
from lip.engine.features import build_features,stack_features
from lip.engine.runtime import sync
from lip.evaluation.metrics import errors,summarize,visibility_bin,constant_velocity
from lip.evaluation.selection import fixed_balanced_subset
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import center_pose,original_pose,angle
from lip.models.tracker import Tracker


def visibility(root,stream,frame,gt,k,mesh,renderer):
    with np.load(Path(root)/stream['relative_dir']/f'labels_{frame:06d}.npz') as z:
        if 'seg' not in z:return None
        seg=z['seg'];target=seg==stream['object_id']
    rd,_=renderer(mesh,gt,k,640)
    mask=rd[0,:seg.shape[0],:seg.shape[1]].cpu().numpy()>0
    if not mask.any():return None
    return float((mask & target).sum()/mask.sum())


def overlay(path,rgb,pred,gt,mesh,k):
    image=(rgb.transpose(1,2,0)*255).astype('uint8').copy()
    pts=np.asarray(mesh['points'])
    for pose,color in [(gt,(0,255,0)),(pred,(255,40,40))]:
        p=pts@pose[:3,:3].T+pose[:3,3];q=p@k.T
        uv=q[:,:2]/np.maximum(q[:,2:],1e-6)
        for (x,y),z in zip(uv.astype(int),p[:,2]):
            if z>0 and 0<=x<image.shape[1] and 0<=y<image.shape[0]:cv2.circle(image,(int(x),int(y)),1,color,-1)
    cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))


@torch.no_grad()
def evaluate(model,c,root,index_root,out,split='val',mode='closed-loop',limit_streams=None,max_frames=None,allow_verified_subset=False,checkpoint_info=None,
             fp_transition=None,stream_ids=None):
    index=Path(index_root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    audit=check_data_gate(index,allow_verified_subset);model.eval();device=next(model.parameters()).device;renderer=Renderer(device)
    streams=[json.loads(x) for x in (index/'streams.jsonl').read_text().splitlines() if json.loads(x)['split']==split]
    streams=fixed_balanced_subset(sorted(streams,key=lambda x:x['stream_id']),limit_streams,lambda s:s['object_id'],c['seed'])
    if stream_ids is not None:
        requested=set(stream_ids)
        if len(requested)!=len(stream_ids):raise ValueError('Duplicate requested streams')
        streams=[s for s in streams if s['stream_id'] in requested]
        if {s['stream_id'] for s in streams}!=requested:raise ValueError('Requested stream not in selected split')
    if fp_transition is not None and mode!='closed-loop':raise ValueError('FP transition requires closed-loop evaluation')
    rows=[];manifest=dict(seed=c['seed'],mode=mode,split=split,split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],
                         initial_pose_source='gt_first_frame' if mode=='closed-loop' else 'noisy_strict_past_gt',
                         full_sequences=max_frames is None,quick_subset=limit_streams is not None,
                         loss_symmetry='canonical_gt',metric_points='all mesh vertices',streams=[s['stream_id'] for s in streams])
    manifest['data_complete']=audit['complete'];manifest['diagnostic_verified_subset']=allow_verified_subset
    manifest['diagnostic_train_subset']=allow_verified_subset and split=='train'
    manifest['checkpoint']=checkpoint_info or {'source':'live model in caller'}
    manifest['config']=c
    manifest.update(method='LIP+FP' if fp_transition is not None else 'LIP',
                    history_state_source='post_FP' if fp_transition is not None else 'LIP',
                    fp_iterations=2 if fp_transition is not None else 0,
                    precision='LIP bf16; official FP float16' if fp_transition is not None else c['precision'])
    thresholds=motion_thresholds(index,c.get('motion_speed_quantile',.75))
    manifest['moving_thresholds']=thresholds
    predictions=(out/'predictions.jsonl').open('w');first_failure={};latencies=[]
    if mode=='one-step':
        ds=ClipDataset(root,index,c['clip_length'],split=split,seed=c['seed'],augmentation=False)
        # Same frozen object-balanced diagnostic population for all methods.
        chosen=fixed_balanced_subset(ds.pool,max_frames,lambda item:ds.streams[item['stream']]['object_id'],c['seed'])
        ds.fixed=chosen;manifest['clips']=chosen
        for i in range(len(chosen)):
            item=ds[i];s=item['stream'];mesh=item['mesh'];l=c['clip_length'];g=torch.Generator().manual_seed(c['seed']+i)
            history,_=noisy_history(item['poses'][:l].to(device),float(mesh['diameter']),g,True)
            base=history[-1];valid=torch.ones(l,dtype=torch.bool,device=device);pv=valid.clone();pv[-1]=False
            hist=torch.cat((history[1:],torch.eye(4,device=device)[None]))
            f,_=build_features(item['rgb'][:l].to(device),item['depth'][:l].to(device),hist,item['times'][:l].to(device),
                               valid,pv,base,item['k'].to(device),mesh,renderer,c['image_size'],c['crop_expansion'])
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=c['precision']=='bf16'):pred=model(**stack_features([f]))['pose_centered'][0]
            cv=constant_velocity(history,item['frames'][:l].to(device)/audit['fps'],item['times'][l-1].to(device))
            gt=item['poses'][l].numpy();frame=int(item['frames'][l]);v=visibility(root,s,frame,torch.from_numpy(gt).to(device),item['k'].to(device),mesh,renderer)
            for name,p in [('zero-motion',base),('constant-velocity',cv),('learned',pred)]:
                e=errors(p.cpu(),gt,mesh['vertices'],float(mesh['diameter']))
                row=dict(**e,method=name,object_id=s['object_id'],stream_id=s['stream_id'],frame_index=frame,
                         visibility_bin=visibility_bin(v),moving=bool(item['sample']['moving']),pose_centered=p.cpu().tolist(),
                         pose_original=original_pose(p,torch.as_tensor(mesh['center'],device=device)).cpu().tolist(),lost=False)
                predictions.write(json.dumps(row)+'\n');rows.append(row)
    else:
        for s in streams:
            with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            with np.load(index/s['pose_cache']) as z:frames=z['frames'].copy();original=z['poses'].copy()
            centered=center_pose(torch.from_numpy(original),torch.from_numpy(mesh['center']))
            k=torch.tensor(s['intrinsics'],device=device);accepted=[];rgbs=[];depths=[];ts=[];streak=0
            for j,frame in enumerate(frames[:max_frames]):
                sync(device);begin=time.perf_counter();rgb,depth=read_frame(root,s,int(frame),audit['depth_scale_to_m'])
                rgbs.append(torch.from_numpy(rgb).to(device));depths.append(torch.from_numpy(depth).to(device));ts.append(float(frame/audit['fps']))
                preprocess=time.perf_counter()-begin;rt=nt=ft=0.;nonfinite_output=False;proposal=None
                if j==0:pred=centered[0].to(device)  # Only allowed GT -> state boundary.
                else:
                    l=c['clip_length'];n=min(l,len(rgbs));pad=l-n
                    r=torch.stack([rgbs[-n]]*pad+rgbs[-n:]);d=torch.stack([depths[-n]]*pad+depths[-n:])
                    hist=torch.stack([accepted[max(0,j-n+1)]]*pad+accepted[max(0,j-n+1):]+[torch.eye(4,device=device)])
                    valid=torch.arange(l,device=device)>=pad;pv=valid.clone();pv[-1]=False
                    t=torch.tensor([ts[-n]]*pad+ts[-n:],device=device)
                    sync(device);start=time.perf_counter()
                    f,_=build_features(r,d,hist,t,valid,pv,accepted[-1],k,mesh,renderer,c['image_size'],c['crop_expansion'])
                    sync(device);rt=time.perf_counter()-start;start=time.perf_counter()
                    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=c['precision']=='bf16'):
                        pred=model(**stack_features([f]))['pose_centered'][0]
                    sync(device);nt=time.perf_counter()-start
                    if not torch.isfinite(pred).all():pred=accepted[-1].clone();nonfinite_output=True
                    if fp_transition is not None:
                        if nonfinite_output:raise FloatingPointError('Nonfinite LIP proposal in hybrid evaluation')
                        proposal=pred.detach().clone();sync(device);start=time.perf_counter()
                        pred=fp_transition(proposal,torch.from_numpy(rgb),torch.from_numpy(depth),k,s['mesh_path'],mesh['center'])
                        if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite refined state')
                        sync(device);ft=time.perf_counter()-start
                accepted.append(pred.detach());sync(device);end_to_end=time.perf_counter()-begin
                # GT becomes available only in this metric branch, after state commit.
                gt=centered[j];v=visibility(root,s,int(frame),gt.to(device),k,mesh,renderer)
                e=errors(pred.cpu(),gt,mesh['vertices'],float(mesh['diameter']))
                streak=streak+1 if e['adds_01']==0 else 0
                if streak==5 and s['stream_id'] not in first_failure:first_failure[s['stream_id']]=dict(frame=int(frame),elapsed_frames=j,seconds=ts[-1]-ts[0])
                moving=False
                if j:
                    dt=(frames[j]-frames[j-1])/audit['fps']
                    moving=bool((gt[:3,3]-centered[j-1,:3,3]).norm()/float(mesh['diameter'])/dt>thresholds['center_d_per_sec'] or
                                angle(gt[:3,:3]@centered[j-1,:3,:3].T)/dt>thresholds['rotation_rad_per_sec'])
                row=dict(**e,object_id=s['object_id'],stream_id=s['stream_id'],frame_index=int(frame),visibility=v,visibility_bin=visibility_bin(v),
                         moving=moving,lost=streak>=5 or nonfinite_output,nonfinite_output=nonfinite_output,pose_centered=pred.cpu().tolist(),pose_original=original_pose(pred,torch.as_tensor(mesh['center'],device=device)).cpu().tolist())
                if proposal is not None:row['lip_proposal_centered']=proposal.cpu().tolist()
                predictions.write(json.dumps(row)+'\n');predictions.flush();rows.append(row)
                if j:latencies.append(dict(preprocess=preprocess,render=rt,network=nt,optional_fp=ft,end_to_end=end_to_end))
                if j<4 or j%30==0 or j==len(frames[:max_frames])-1 or streak==5:overlay(out/(s['stream_id'].replace('/','_')+f'_{frame:06d}.jpg'),rgb,pred.cpu().numpy(),gt.numpy(),mesh,np.array(s['intrinsics']))
                # Bound observation memory; accepted poses are small and retained for audit.
                if len(rgbs)>c['clip_length']:rgbs.pop(0);depths.pop(0)
            print(json.dumps(dict(completed_stream=s['stream_id'],frames=len(rows))),flush=True)
    predictions.close();manifest['first_five_consecutive_adds_failures']=first_failure
    if mode=='one-step':report={m:summarize([r for r in rows if r['method']==m]) for m in ['zero-motion','constant-velocity','learned']}
    else:report=summarize(rows)
    report['latency_seconds']={k:dict(mean=float(np.mean([x[k] for x in latencies])),p95=float(np.quantile([x[k] for x in latencies],.95))) for k in latencies[0]} if latencies else {}
    report['latency_scope']='batch=1 synchronized; excludes GT metrics and visualization; includes raw decode and lazy FP object setup when enabled'
    manifest.update(completed=True,frames=len(rows))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'metrics.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',default=os.getenv('DEX_YCB_DIR'));p.add_argument('--index-root',default='cache/dexycb_s0')
    p.add_argument('--out',required=True);p.add_argument('--split',choices=['train','val','test'],default='val');p.add_argument('--mode',choices=['one-step','closed-loop'],default='closed-loop')
    p.add_argument('--limit-streams',type=int);p.add_argument('--max-frames',type=int);p.add_argument('--test-finalized',action='store_true')
    p.add_argument('--allow-verified-subset',action='store_true');a=p.parse_args()
    if a.split=='train' and not a.allow_verified_subset:p.error('Train evaluation is only a verified-subset diagnostic')
    if a.split=='test' and not a.test_finalized:p.error('Test requires --test-finalized; do not tune on test')
    if not a.data_root:p.error('DEX_YCB_DIR is required')
    c=load_config(a.config);audit=check_data_gate(a.index_root,a.allow_verified_subset)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck['split_hash']!=audit['split_hash'] or ck['mesh_hash']!=audit['mesh_hash']:raise ValueError('Checkpoint data mismatch')
    device='cuda' if torch.cuda.is_available() else 'cpu';model=Tracker(False).to(device);model.load_state_dict(ck['model'])
    h=hashlib.sha256()
    with open(a.checkpoint,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    info=dict(path=str(Path(a.checkpoint).resolve()),sha256=h.hexdigest(),global_step=ck['global_step'],code_sha256=ck.get('code_sha256'))
    evaluate(model,c,a.data_root,a.index_root,a.out,a.split,a.mode,a.limit_streams,a.max_frames,a.allow_verified_subset,info)


if __name__=='__main__':main()
