"""Actual RGB and full trajectories for stable-entry occlusion failures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull,QhullError


def hull(vertices,pose,k):
    camera=vertices@pose[:3,:3].T+pose[:3,3];camera=camera[camera[:,2]>.01];uv=camera@k.T;xy=uv[:,:2]/uv[:,2:]
    if len(xy)<3:return None
    try:boundary=xy[ConvexHull(xy).vertices]
    except QhullError:return None
    return np.concatenate((boundary,boundary[:1]))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--events',required=True,type=Path);p.add_argument('--evaluation',action='append',required=True)
    for name in ('index-root','data-root','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);report=json.loads(a.events.read_text());reference=report['reference'];rows={}
    for arg in a.evaluation:
        name,path=arg.split('=',1);raw=(Path(path)/'predictions.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==report['prediction_sha256'][name]
        rows[name]={(r['stream_id'],r['frame_index']):r for r in map(json.loads,raw.splitlines())}
    assert set(rows)==set(report['checkpoint_sha256'])
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    selected=[];seen=set()
    for e in sorted([e for e in report['events'] if e['reference_entry']=='stable_good' and not e['arms'][reference]['end_success'] and e['length']>8],key=lambda e:e['arms'][reference]['max_center_mm'],reverse=True):
        physical='/'.join(e['stream_id'].split('/')[:2])
        if physical in seen:continue
        selected.append(e);seen.add(physical)
        if len(selected)==4:break
    output=[];colors=dict(zip(rows,('tab:blue','tab:orange','tab:red','tab:green','tab:purple')))
    if len(colors)!=len(rows):raise ValueError('At most five comparison arms supported')
    for number,e in enumerate(selected):
        sid=e['stream_id'];s=streams[sid];diameter=s['mesh_diameter'];k=np.array(s['intrinsics'])
        with np.load(a.index_root/s['mesh_cache']) as z:vertices=z['vertices'].copy();center=z['center'].copy()
        with np.load(a.index_root/s['pose_cache']) as z:gt={int(f):pose.copy() for f,pose in zip(z['frames'],z['poses'])}
        for pose in gt.values():pose[:3,3]+=pose[:3,:3]@center
        frames=[e['pre_frames'][-1],e['frames'][len(e['frames'])//2],e['end'],e['post_frames'][-1] if e['post_frames'] else e['frames'][len(e['frames'])//4]]
        labels=['Before occlusion','Mid occlusion','Occlusion end','Post window end' if e['post_frames'] else 'Earlier occlusion (post censored)']
        fig=plt.figure(figsize=(16,9));grid=fig.add_gridspec(3,4,height_ratios=[2.3,1,1])
        for col,(frame,label) in enumerate(zip(frames,labels)):
            ax=fig.add_subplot(grid[0,col]);rgb=np.asarray(Image.open(a.data_root/s['relative_dir']/f'color_{frame:06d}.jpg').convert('RGB'));ax.imshow(rgb)
            for name,pose,color in [('GT',gt[frame],'lime')]+[(n,np.array(rs[(sid,frame)]['pose_centered']),colors[n]) for n,rs in rows.items()]:
                boundary=hull(vertices,pose,k)
                if boundary is not None:ax.plot(boundary[:,0],boundary[:,1],color=color,lw=1,label=name)
            ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0);ax.axis('off');ax.set_title(f"{label}, frame {frame}\nvisibility={rows[reference][(sid,frame)]['visibility']:.3f}",fontsize=9)
            if col==0:ax.legend(fontsize=6,loc='lower left')
        chronological=sorted(f for stream,f in rows[reference] if stream==sid)
        ax=fig.add_subplot(grid[1,:2]);center_ax=fig.add_subplot(grid[1,2:]);rot_ax=fig.add_subplot(grid[2,:2]);vis_ax=fig.add_subplot(grid[2,2:])
        for name,rs in rows.items():
            trajectory=[rs[(sid,f)] for f in chronological]
            ax.plot(chronological,[v['adds_m']/diameter for v in trajectory],color=colors[name],label=name)
            center_ax.plot(chronological,[v['center_mm'] for v in trajectory],color=colors[name])
            rot_ax.plot(chronological,[v['rotation_deg'] for v in trajectory],color=colors[name])
        ax.axhline(.05,color='black',ls='--',lw=.6);ax.set_ylabel('ADD-S / d');ax.legend(fontsize=8);center_ax.set_ylabel('Center error (mm)');rot_ax.set_ylabel('Canonical rotation error (deg)')
        ref=[rows[reference][(sid,f)] for f in chronological];vis_ax.plot(chronological,[v['visibility'] if v['visibility'] is not None else np.nan for v in ref],label='GT visibility proxy')
        vis_ax.plot(chronological,[v.get('observation_support',np.nan) for v in ref],label='Parent depth support');vis_ax.legend(fontsize=8);vis_ax.set_ylim(-.05,1.05)
        for axis in (ax,center_ax,rot_ax,vis_ax):axis.axvspan(e['start']-.5,e['end']+.5,color='grey',alpha=.15);axis.set_xlabel('Frame')
        fig.suptitle(sid+'\nFixed reference had three accurate clear frames before occlusion. RGB plus mesh convex hulls, not visible-surface masks.',fontsize=10)
        fig.tight_layout();path=a.out/f'event_{number:02d}.png';fig.savefig(path,dpi=130);plt.close(fig)
        output.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stream_id=sid,object_id=e['object_id'],start=e['start'],end=e['end'],frames=frames,censor_reason=e['censor_reason'],arms=e['arms']))
    (a.out/'figures.json').write_text(json.dumps(dict(completed=True,selection='Top max-center-error long episodes with stable-good reference entry and failed reference end, unique physical sequence; four cases',
        events_sha256=hashlib.sha256(a.events.read_bytes()).hexdigest(),visual_review_completed=False,human_verified=False,figures=output),indent=2))


if __name__=='__main__':main()
