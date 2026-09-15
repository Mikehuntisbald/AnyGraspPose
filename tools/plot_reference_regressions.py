"""Inspect fixed high-visibility threshold regressions and an improvement example."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_pose_robustness import stable_success_position
from plot_occlusion_event_cases import hull


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('audit','reference-eval','candidate-eval','index-root','data-root','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--reference-label',default='M1A1');p.add_argument('--candidate-label',default='b04')
    p.add_argument('--rotation-error',action='store_true',help='Show canonical rotation errors instead of read coefficients')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if len({'GT','Initializer',a.reference_label,a.candidate_label})!=4:raise ValueError('Distinct plot labels required')
    audit=json.loads((a.audit/'analysis.json').read_text());changed=list(map(json.loads,(a.audit/'changed_frames.jsonl').read_text().splitlines()))
    evaluations={};manifests={}
    for name,folder in [('reference',a.reference_eval),('candidate',a.candidate_eval)]:
        raw=(folder/'predictions.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==audit['prediction_sha256'][name]
        evaluations[name]={(r['stream_id'],r['frame_index']):r for r in map(json.loads,raw.splitlines())}
        manifests[name]=json.loads((folder/'manifest.json').read_text())
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    groups=defaultdict(list)
    for row in changed:
        if row['visibility'] is not None and row['visibility']>=.5:
            groups['regression' if row['reference_adds005'] else 'improvement',row['stream_id']].append(row)
    selected=[]
    for kind,count in [('regression',3),('improvement',1)]:
        seen=set();objects=set()
        for (label,sid),rows in sorted(groups.items(),key=lambda item:(-len(item[1]),-max(abs(r['candidate']['adds_percent_d']-r['reference']['adds_percent_d']) for r in item[1]),item[0])):
            physical='/'.join(sid.split('/')[:2]);oid=rows[0]['object_id']
            if label!=kind or physical in seen or oid in objects:continue
            selected.append((kind,sid,rows));seen.add(physical);objects.add(oid)
            if len(seen)==count:break
    colors={'GT':'lime',a.reference_label:'tab:blue',a.candidate_label:'tab:red','Initializer':'deepskyblue'};figures=[]
    for number,(kind,sid,changes) in enumerate(selected):
        stream=streams[sid];assert stream['split']=='val';k=np.asarray(stream['intrinsics'])
        with np.load(a.index_root/stream['mesh_cache']) as z:vertices=z['vertices'].copy();center=z['center'].copy();diameter=float(z['diameter'])
        with np.load(a.index_root/stream['pose_cache']) as z:gt={int(f):pose.copy() for f,pose in zip(z['frames'],z['poses'])}
        for pose in gt.values():pose[:3,3]+=pose[:3,:3]@center
        frames=sorted(f for s,f in evaluations['reference'] if s==sid)
        trajectories={name:[data[sid,f] for f in frames] for name,data in evaluations.items()}
        initial=np.asarray(trajectories['candidate'][0]['pose_centered']);assert trajectories['reference'][0]['pose_centered']==initial.tolist()
        stable=[stable_success_position(rs) for rs in trajectories.values()]
        recovered=max(stable) if all(v is not None for v in stable) else 1
        peak=max(changes,key=lambda r:abs(r['candidate']['adds_percent_d']-r['reference']['adds_percent_d']))['frame']
        chosen=[frames[0],frames[recovered],peak,frames[min(frames.index(peak)+8,len(frames)-1)]]
        fig=plt.figure(figsize=(16,12));grid=fig.add_gridspec(4,4,height_ratios=[2,1.5,1,1])
        for col,frame in enumerate(chosen):
            rgb=np.asarray(Image.open(a.data_root/stream['relative_dir']/f'color_{frame:06d}.jpg').convert('RGB'))
            depth=np.asarray(Image.open(a.data_root/stream['relative_dir']/f'aligned_depth_to_color_{frame:06d}.png'))
            poses={'GT':gt[frame],a.reference_label:np.asarray(evaluations['reference'][sid,frame]['pose_centered']),
                a.candidate_label:np.asarray(evaluations['candidate'][sid,frame]['pose_centered']),'Initializer':initial}
            bounds={n:hull(vertices,pose,k) for n,pose in poses.items()};valid=[b for b in bounds.values() if b is not None]
            points=np.concatenate(valid) if valid else np.array([[0,0],[rgb.shape[1]-1,rgb.shape[0]-1]])
            x0=max(0,int(points[:,0].min())-35);x1=min(rgb.shape[1],int(points[:,0].max())+35)
            y0=max(0,int(points[:,1].min())-35);y1=min(rgb.shape[0],int(points[:,1].max())+35)
            if x1<=x0 or y1<=y0:x0,y0,x1,y1=0,0,rgb.shape[1],rgb.shape[0]
            ax=fig.add_subplot(grid[0,col]);ax.imshow(rgb)
            dax=fig.add_subplot(grid[1,col]);d=depth[y0:y1,x0:x1].astype(float);nz=d[d>0];lo,hi=np.quantile(nz,[.02,.98]) if len(nz) else (0,1)
            dax.imshow(np.ma.masked_where(d<=0,d),cmap='viridis',vmin=lo,vmax=hi,extent=(x0,x1,y1,y0))
            for name,boundary in bounds.items():
                if boundary is not None:
                    for axis in (ax,dax):axis.plot(boundary[:,0],boundary[:,1],color=colors[name],lw=1,label=name)
            for axis in (ax,dax):axis.set_xlim(x0,x1);axis.set_ylim(y1,y0);axis.axis('off')
            v=evaluations['candidate'][sid,frame]['visibility'];row=evaluations['candidate'][sid,frame]
            ax.set_title(f'Frame {frame}; visibility={v if v is None else round(v,3)}\nsupport={row.get("observation_support",float("nan")):.3f}',fontsize=9)
            dax.set_title('Observed depth zoom, raw units',fontsize=8)
            if col==0:ax.legend(fontsize=7,loc='lower left')
        axes=[fig.add_subplot(grid[2,:2]),fig.add_subplot(grid[2,2:]),fig.add_subplot(grid[3,:2]),fig.add_subplot(grid[3,2:])]
        for name,rs in trajectories.items():
            label=a.reference_label if name=='reference' else a.candidate_label;color=colors[label]
            axes[0].plot(frames,[r['adds_m']/diameter for r in rs],label=label,color=color)
            axes[1].plot(frames,[r['center_mm'] for r in rs],label=label,color=color)
        axes[0].axhline(.05,ls='--',color='black',lw=.7);axes[0].set_ylabel('ADD-S / d');axes[0].legend(fontsize=8)
        axes[1].set_ylabel('Center error (mm)')
        rs=trajectories['candidate']
        if a.rotation_error:
            for name,trajectory in trajectories.items():
                label=a.reference_label if name=='reference' else a.candidate_label
                axes[2].plot(frames,[r['rotation_deg'] for r in trajectory],label=label,color=colors[label])
            axes[2].set_ylabel('Canonical rotation error (deg)')
        else:
            for key,label in [('reference_rotation_coefficient','Rotation coefficient'),('reference_center_coefficient','Center coefficient')]:
                axes[2].plot(frames,[r.get(key,np.nan) for r in rs],label=label)
            axes[2].set_ylabel('Signed reference coefficient')
        axes[2].legend(fontsize=8)
        axes[3].plot(frames,[r['visibility'] if r['visibility'] is not None else np.nan for r in rs],label='GT visibility')
        axes[3].plot(frames,[r.get('observation_support',np.nan) for r in rs],label='Depth support');axes[3].set_ylim(-.05,1.05);axes[3].legend(fontsize=8)
        for axis in axes:
            for change in changes:axis.axvspan(change['frame']-.45,change['frame']+.45,color='grey',alpha=.15)
            axis.axvline(peak,color='black',ls=':',lw=.7);axis.set_xlabel('Frame; shaded: selected threshold changes')
        fig.suptitle(sid+'\n'+kind+' at visibility >= 0.5; diagnostic projections are convex hulls, not visible masks. GT is posthoc only.',fontsize=10)
        fig.tight_layout();path=a.out/f'case_{number:02d}.png';fig.savefig(path,dpi=135);plt.close(fig)
        figures.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kind=kind,stream_id=sid,object_id=stream['object_id'],frames=chosen,
            changed_frames=[r['frame'] for r in changes],peak=peak,stable_success_positions=stable))
    regressions=[r for r in changed if r['reference_adds005']]
    result=dict(completed=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_audit_sha256=hashlib.sha256((a.audit/'analysis.json').read_bytes()).hexdigest(),
        prediction_sha256=audit['prediction_sha256'],checkpoint_sha256=audit['checkpoints'],
        labels={'reference':a.reference_label,'candidate':a.candidate_label},rotation_error_panel=a.rotation_error,
        selection='Three distinct-object and physical-sequence high-visibility regression streams plus one improvement stream, sorted by qualifying frame count, absolute ADD-S error change, and stream ID. Peak is maximum absolute error change within that fixed group. All bad-initial threshold changes remain in the input audit.',
        regression_visibility_counts=dict(Counter('unknown' if r['visibility'] is None else 'lt03' if r['visibility']<.3 else '03_to_05' if r['visibility']<.5 else 'ge05' for r in regressions)),
        figures=figures,visual_review_completed=False,human_verified=False)
    (a.out/'figures.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
