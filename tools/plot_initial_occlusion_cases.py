"""Inspect both static drift and moving-object counterexamples to holding a pose."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_occlusion_event_cases import hull


def select_cases(rows, per_kind=2):
    groups = defaultdict(list)
    for row in rows:
        if row['visibility'] >= .3: continue
        if row['initial_stationary_prefix'] and row['hold_initial_success'] and not row['lip_success']:
            kind = 'hold_helps_static'
        elif not row['initial_stationary_prefix'] and row['lip_success'] and not row['hold_initial_success']:
            kind = 'lip_helps_motion'
        else: continue
        groups[kind, row['stream_id']].append(row)
    selected=[]
    for kind in ('hold_helps_static', 'lip_helps_motion'):
        seen_physical=set();seen_objects=set()
        for (category, sid), rr in sorted(groups.items(), key=lambda kv:(-len(kv[1]),kv[0])):
            if category != kind or len(rr) < 3: continue
            physical='/'.join(sid.split('/')[:2]);obj=rr[0]['object_id']
            if physical in seen_physical or obj in seen_objects: continue
            seen_physical.add(physical);seen_objects.add(obj)
            selected.append(dict(kind=kind,stream_id=sid,object_id=obj,qualifying_frames=sorted(r['frame'] for r in rr)))
            if len(seen_physical) == per_kind: break
    return selected, {kind: dict(frames=sum(len(v) for (c,_),v in groups.items() if c==kind),
        streams=sum(c==kind for c,_ in groups),objects=dict(Counter(r['object_id'] for (c,_),v in groups.items() if c==kind for r in v)))
        for kind in ('hold_helps_static','lip_helps_motion')}


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('runtime','hold-audit','evaluation','index-root','data-root','out'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--comparison',action='append',default=[],help='Optional NAME=completed_evaluation, at most three')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.runtime/'src'))
    from lip.evaluation.metrics import errors
    from lip.geometry.so3 import center_pose,angle
    torch.set_num_threads(1)
    audit=json.loads((a.hold_audit/'audit.json').read_text());raw=(a.evaluation/'predictions.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==audit['prediction_sha256']
    hold_raw=(a.hold_audit/'frames.jsonl').read_bytes();hold_rows=list(map(json.loads,hold_raw.splitlines()))
    selected,populations=select_cases(hold_rows);data=defaultdict(list)
    for r in map(json.loads,raw.splitlines()):data[r['stream_id']].append(r)
    base_manifest=json.loads((a.evaluation/'manifest.json').read_text())
    base_rows={(r['stream_id'],r['frame_index']):r for rows in data.values() for r in rows}
    comparisons={};comparison_meta={}
    if len(a.comparison)>3:raise ValueError('At most three named comparisons')
    for value in a.comparison:
        name,path=value.split('=',1);folder=Path(path)
        if name in comparisons or name in ('GT','LIP','Held initializer'):raise ValueError('Distinct comparison labels required')
        m=json.loads((folder/'manifest.json').read_text());candidate_raw=(folder/'predictions.jsonl').read_bytes()
        rows=list(map(json.loads,candidate_raw.splitlines()));keyed={(r['stream_id'],r['frame_index']):r for r in rows}
        assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320 and m['split']=='val'
        assert m['fp_calls']==m['critic_calls']==0 and all(m[k]==base_manifest[k] for k in ('split_hash','mesh_hash','initial_poses_sha256'))
        assert len(keyed)==len(rows)==23200 and set(keyed)==set(base_rows)
        assert all(keyed[k]['visibility']==r['visibility'] for k,r in base_rows.items())
        assert all(keyed[k]['pose_centered']==r['pose_centered'] for k,r in base_rows.items() if r['initialization'])
        comparisons[name]=keyed;comparison_meta[name]=dict(checkpoint_sha256=m['checkpoint_sha256'],prediction_sha256=hashlib.sha256(candidate_raw).hexdigest())
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    figures=[];chronology=[]
    for number,case in enumerate(selected):
        sid=case['stream_id'];s=streams[sid];assert s['split']=='val'
        trajectory=sorted(data[sid],key=lambda r:r['frame_index']);lookup={r['frame_index']:r for r in trajectory}
        with np.load(a.index_root/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
        with np.load(a.index_root/s['pose_cache']) as z:frames=z['frames'].copy();gt=center_pose(torch.from_numpy(z['poses'].copy()),torch.from_numpy(mesh['center'])).numpy()
        assert [int(f) for f in frames]==[r['frame_index'] for r in trajectory]
        diameter=float(mesh['diameter']);k=np.asarray(s['intrinsics']);initial=np.asarray(trajectory[0]['pose_centered'],dtype='f4')
        hold_errors=[errors(initial,g,mesh['vertices'],diameter,dtype='f4') for g in gt]
        gt_rotation=(angle(torch.from_numpy(gt[:,:3,:3])@torch.from_numpy(gt[0,:3,:3]).T)*180/torch.pi).numpy()
        gt_center=np.linalg.norm(gt[:,:3,3]-gt[0,:3,3],axis=-1)/diameter
        gt_at=dict(zip(map(int,frames),gt));positions={int(f):i for i,f in enumerate(frames)}
        chosen=[int(frames[0]),case['qualifying_frames'][0],case['qualifying_frames'][len(case['qualifying_frames'])//2],case['qualifying_frames'][-1]]
        fig=plt.figure(figsize=(16,12));grid=fig.add_gridspec(4,4,height_ratios=[2,1.5,1,1])
        colors={'GT':'lime','LIP':'tab:red','Held initializer':'deepskyblue',**dict(zip(comparisons,('tab:blue','tab:orange','tab:purple')))}
        for col,frame in enumerate(chosen):
            rgb=np.asarray(Image.open(a.data_root/s['relative_dir']/f'color_{frame:06d}.jpg').convert('RGB'))
            depth=np.asarray(Image.open(a.data_root/s['relative_dir']/f'aligned_depth_to_color_{frame:06d}.png'))
            poses={'GT':gt_at[frame],'LIP':np.asarray(lookup[frame]['pose_centered']),'Held initializer':initial}
            poses.update({name:np.asarray(rows[sid,frame]['pose_centered']) for name,rows in comparisons.items()})
            ax=fig.add_subplot(grid[0,col]);ax.imshow(rgb)
            boundaries={name:hull(mesh['vertices'],pose,k) for name,pose in poses.items()}
            for name,boundary in boundaries.items():
                if boundary is not None:ax.plot(boundary[:,0],boundary[:,1],color=colors[name],lw=1,label=name)
            ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0);ax.axis('off')
            visibility=lookup[frame]['visibility'];v='unknown' if visibility is None else f'{visibility:.3f}'
            ax.set_title(f'Frame {frame}; visibility {v}',fontsize=9)
            if col==0:ax.legend(fontsize=6,loc='lower left')
            # Fixed diagnostic zoom around the union of GT/actor/prior projections.
            valid=[b for b in boundaries.values() if b is not None]
            points=np.concatenate(valid) if valid else np.array([[0,0],[rgb.shape[1]-1,rgb.shape[0]-1]])
            x0=max(0,int(np.floor(points[:,0].min()))-20);x1=min(rgb.shape[1],int(np.ceil(points[:,0].max()))+20)
            y0=max(0,int(np.floor(points[:,1].min()))-20);y1=min(rgb.shape[0],int(np.ceil(points[:,1].max()))+20)
            if x1<=x0 or y1<=y0:x0,y0,x1,y1=0,0,rgb.shape[1],rgb.shape[0]
            dax=fig.add_subplot(grid[1,col]);values=depth[y0:y1,x0:x1].astype(float)
            nonzero=values[values>0];lo,hi=np.quantile(nonzero,[.02,.98]) if len(nonzero) else (0,1)
            dax.imshow(np.ma.masked_where(values<=0,values),cmap='viridis',vmin=lo,vmax=hi,extent=(x0,x1,y1,y0))
            for name,boundary in boundaries.items():
                if boundary is not None:dax.plot(boundary[:,0],boundary[:,1],color=colors[name],lw=1)
            dax.set_xlim(x0,x1);dax.set_ylim(y1,y0);dax.axis('off');dax.set_title('Observed depth zoom (raw units)',fontsize=8)
        axs=[fig.add_subplot(grid[2,:2]),fig.add_subplot(grid[2,2:]),fig.add_subplot(grid[3,:2]),fig.add_subplot(grid[3,2:])]
        for index,(field,label,scale) in enumerate([('adds_m','ADD-S / d',1/diameter),('center_mm','Center error (mm)',1),('rotation_deg','Canonical rotation error (deg)',1)]):
            axs[index].plot(frames,[r[field]*scale for r in trajectory],label='LIP',color=colors['LIP'])
            axs[index].plot(frames,[r[field]*scale for r in hold_errors],label='Held initializer',color=colors['Held initializer'])
            for name,rows in comparisons.items():axs[index].plot(frames,[rows[sid,int(frame)][field]*scale for frame in frames],label=name,color=colors[name])
            axs[index].set_ylabel(label)
        axs[0].axhline(.05,color='black',lw=.7,ls='--');axs[0].legend(fontsize=8)
        axs[3].plot(frames,[r['visibility'] if r['visibility'] is not None else np.nan for r in trajectory],label='GT visibility')
        axs[3].plot(frames,[r.get('observation_support',np.nan) for r in trajectory],label='LIP depth support')
        for name,rows in comparisons.items():
            if any('reference_write_center_coefficient' in rows[sid,int(f)] for f in frames):
                axs[3].plot(frames,[rows[sid,int(f)].get('reference_write_center_coefficient',np.nan) for f in frames],label=name+' center write',color=colors[name],ls='--')
        axs[3].set_ylim(-.02,1.02);axs[3].legend(fontsize=8)
        for ax in axs:
            ax.axvspan(0,8,color='grey',alpha=.15);ax.set_xlabel('Frame; shaded: first 8 updates')
        first_failure=next((r['frame_index'] for r in trajectory if not r['initialization'] and (not r['adds_005'] or r['status']!='ok')),None)
        case=dict(case,first_actor_failure=first_failure,initial_visibility=trajectory[0]['visibility'],first_update_center_mm=trajectory[1]['center_mm'],
            first_update_rotation_deg=trajectory[1]['rotation_deg'],qualifying_max_gt_rotation_deg=max(float(gt_rotation[positions[f]]) for f in case['qualifying_frames']),
            qualifying_max_gt_center_d=max(float(gt_center[positions[f]]) for f in case['qualifying_frames']),frames=chosen)
        case['comparison_summary']={name:dict(first_failure=next((int(f) for f in frames if not rows[sid,int(f)]['initialization'] and
            (not rows[sid,int(f)]['adds_005'] or rows[sid,int(f)]['status']!='ok')),None),
            qualifying_successes=sum(bool(rows[sid,f]['adds_005']) and rows[sid,f]['status']=='ok' for f in case['qualifying_frames']),
            qualifying_frames=len(case['qualifying_frames'])) for name,rows in comparisons.items()}
        fig.suptitle(sid+'\n'+case['kind']+'; selected posthoc examples, not an aggregate improvement claim. GT/reference are diagnostic only.',fontsize=10)
        fig.tight_layout();path=a.out/f'case_{number:02d}.png';fig.savefig(path,dpi=140);plt.close(fig)
        figures.append(dict(case,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for i,r in enumerate(trajectory):chronology.append(dict(stream_id=sid,frame=r['frame_index'],lip_adds_d=r['adds_m']/diameter,hold_adds_d=hold_errors[i]['adds_m']/diameter,gt_rotation_from_initial_deg=float(gt_rotation[i]),gt_center_from_initial_d=float(gt_center[i])))
    (a.out/'trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in chronology))
    report=dict(completed=True,prediction_sha256=audit['prediction_sha256'],checkpoint_sha256=audit['checkpoint_sha256'],
        hold_frames_sha256=hashlib.sha256(hold_raw).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='Two distinct-object/physical-sequence cases per condition, sorted by number of qualifying severe-occlusion frames (at least three), stream ID tie-break. Static cases require hold success/LIP failure; moving cases require the reverse. No new model or selected score.',
        populations=populations,figures=figures,visual_review_completed=False,human_verified=False)
    report['comparisons']=comparison_meta
    (a.out/'figures.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
