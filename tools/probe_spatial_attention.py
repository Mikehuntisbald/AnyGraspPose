"""Measure past foreground attention without changing any tracking outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('runtime','experiment','streams','index-root','data-root','out'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.runtime/'src'))
    from lip.engine.stream_config import load_stream_config,make_model
    from lip.engine.stream_checkpoint import load_init,sha,source_hash
    from lip.engine.config import check_data_gate
    from lip.geometry.renderer import Renderer
    from lip.data.index import read_frame
    torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False);e=json.loads((a.experiment/'experiment.json').read_text());c=load_stream_config(e['arms']['spatial']['config'])
    checkpoint=a.experiment/'spatial/train/last.pt';evaluation=a.experiment/'spatial/s0_val';manifest=json.loads((evaluation/'manifest.json').read_text())
    assert sha(checkpoint)==manifest['checkpoint_sha256'] and source_hash()==manifest['source_sha256']
    model=make_model(c).cuda().eval();load_init(checkpoint,model,check_data_gate(a.index_root),c);renderer=Renderer('cuda')
    selected=json.loads(a.streams.read_text());assert len(selected)==len(set(selected))
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())};assert all(streams[s]['split']=='val' for s in selected)
    saved={(r['stream_id'],r['frame_index']):r for r in map(json.loads,(evaluation/'predictions.jsonl').read_text().splitlines()) if r['stream_id'] in selected}
    initials=json.loads(Path(e['initial_poses']).read_text());audit=json.loads((a.index_root/'audit.json').read_text());original=model.patch_read;foreground={};records=[];current_sid=None
    def instrument(objects,bank,patches,quality,features,meta):
        result=original(objects,bank,patches,quality,features,meta)
        frame=int(meta.frame_id[0]);foreground[frame]=F.adaptive_avg_pool2d(features['geometry'][:,3:4].float(),model.dense_side).flatten(1)[0].detach()
        age=meta.frame_id[:,None]-bank.frame_id
        valid=bank.valid&(age>=model.memory_frames)&(age<=model.max_age)&(bank.stream_tag==meta.stream_tag[:,None])&(bank.timestamp<=meta.timestamp[:,None])
        if not bool(valid.any()):return result
        n=model.dense_side**2;allowed=valid.repeat_interleave(n,1)
        condition=model.pose_condition(bank,features,meta);context=(torch.where(bank.valid[:,:,None,None],patches,0.)+condition[:,:,None]).flatten(1,2)
        bias=quality.flatten(1).clamp_min(.25).log().masked_fill(~allowed,float('-inf'))[:,None].repeat_interleave(8,0).to(objects.dtype)
        h,weights=model.patch_attention(objects[:,None],context,context,attn_mask=bias,need_weights=True,average_attn_weights=False)
        h=h[:,0];gate=torch.sigmoid(model.patch_gate(torch.cat((objects,h),-1)));explicit=gate*h
        fg=torch.stack([foreground[int(f)] if bool(v) else torch.zeros_like(foreground[frame]) for f,v in zip(bank.frame_id[0],bank.valid[0])]).flatten()
        fg_tokens=fg>0;prob=weights[0,:,0].float();used=allowed[0]
        entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)/used.sum().float().log()
        records.append(dict(stream_id=current_sid,frame=frame,foreground_token_fraction=float(fg_tokens[used].float().mean()),
            foreground_attention_mass=float(prob[:,fg_tokens].sum(-1).mean()),area_weighted_foreground_mass=float((prob*fg).sum(-1).mean()),
            normalized_entropy=float(entropy.mean()),max_attention=float(prob.max(-1).values.mean()),
            explicit_weight_backend_residual_max_difference=float((explicit.float()-result[0].float()).abs().max()),
            anchor_frames=bank.frame_id[0,valid[0]].cpu().tolist(),usable_tokens=int(used.sum())))
        return result
    model.patch_read=instrument;frames=0
    with torch.no_grad():
        for current_sid in selected:
            foreground.clear();s=streams[current_sid]
            with np.load(a.index_root/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            with np.load(a.index_root/s['pose_cache']) as z:indices=z['frames'].copy();times=z['timestamps'].copy() if 'timestamps' in z else indices.astype('f8')/audit['fps']
            state=model.initialize(initials['poses'][current_sid],mesh,s['intrinsics'],current_sid,times[0],object_id=s['object_id'],camera_id=s['camera_serial'],mesh_hash=sha(a.index_root/s['mesh_cache']))
            assert state.pose_centered.cpu().tolist()==saved[(current_sid,int(indices[0]))]['pose_centered'];frames+=1
            for j,index in enumerate(indices[1:],1):
                rgb,depth=read_frame(a.data_root,s,int(index),audit['depth_scale_to_m'])
                proposal,next_state=model.step(torch.from_numpy(rgb),torch.from_numpy(depth),times[j],state,renderer=renderer,precision=c['precision'],image_size=c['image_size'],crop_expansion=c['crop_expansion'])
                assert proposal['status']=='ok';state=model.commit(proposal,next_state)
                assert proposal['pose_centered'].cpu().tolist()==saved[(current_sid,int(index))]['pose_centered'],(current_sid,int(index),'instrumentation changed pose')
                frames+=1
    for r in records:
        original_row=saved[(r['stream_id'],r['frame'])];r.update(visibility=original_row['visibility'],adds_005=bool(original_row['adds_005']))
    summaries={}
    for name,rs in [('all',records),('low_visibility',[r for r in records if r['visibility'] is not None and r['visibility']<.5]),('low_visibility_failed',[r for r in records if r['visibility'] is not None and r['visibility']<.5 and not r['adds_005']])]:
        summaries[name]=dict(frames=len(rs),**{key:float(np.mean([r[key] for r in rs])) if rs else None for key in ('foreground_token_fraction','foreground_attention_mass','area_weighted_foreground_mass','normalized_entropy','max_attention')})
    result=dict(passed=True,scope='Selected development-val diagnostic. Predictor uses original attention output; explicit weights are computed additionally and may differ by backend numerics. Every full trajectory is bitwise equal to saved evaluation. Foreground denotes the stored predicted-pose rendered silhouette, not GT segmentation or proven correct correspondence.',
        checkpoint_sha256=sha(checkpoint),source_sha256=source_hash(),streams=selected,frames_verified=frames,
        explicit_weight_backend_max_difference=max(r['explicit_weight_backend_residual_max_difference'] for r in records),summaries=summaries)
    (a.out/'attention.json').write_text(json.dumps(result,indent=2,allow_nan=False));(a.out/'frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
