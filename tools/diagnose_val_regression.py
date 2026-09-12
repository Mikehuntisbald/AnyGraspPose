"""Read-only paired validation diagnosis; no training or causal attribution."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
run=root/'runs/lip_v1_s0';out=root/'runs/basin_speed_v1/diagnostics';out.mkdir(exist_ok=True,parents=True)
reports={};rows={};manifests={}
for step in (22000,24000):
    p=run/f'val_{step}'
    reports[step]=json.loads((p/'metrics.json').read_text())
    manifests[step]=json.loads((p/'manifest.json').read_text())
    rows[step]=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
    assert len(rows[step])==len({(r['stream_id'],r['frame_index']) for r in rows[step]})
keys=('split','mode','initial_pose_source','split_hash','mesh_hash','streams','quick_subset')
assert all(manifests[22000][k]==manifests[24000][k] for k in keys)
assert {(r['stream_id'],r['frame_index']) for r in rows[22000]}=={(r['stream_id'],r['frame_index']) for r in rows[24000]}
names=['master_chef_can','cracker_box','sugar_box','tomato_soup_can','mustard_bottle','tuna_fish_can','pudding_box','gelatin_box']
objects=[]
drop=100*(reports[22000]['macro_object']['add_01']-reports[24000]['macro_object']['add_01'])
for oid,name in enumerate(names,1):
    a,b=[reports[s]['per_object_id'][str(oid)] for s in (22000,24000)]
    contribution=100*(a['add_01']['mean']-b['add_01']['mean'])/8
    objects.append(dict(object_id=oid,name=name,frames=a['count'],
                        before={k:a[k]['mean'] for k in ('add_01','adds_01','rotation_deg','center_mm')},
                        after={k:b[k]['mean'] for k in ('add_01','adds_01','rotation_deg','center_mm')},
                        macro_add_drop_pp=contribution,fraction_of_total_drop=contribution/drop))
gelatin={s:[r for r in rows[s] if r['object_id']==8] for s in rows}
late={s:[r for r in rr if r['frame_index']>=46] for s,rr in gelatin.items()}
result=dict(scope='Matched saved pure-LIP closed-loop predictions; descriptive diagnosis, not a treatment-effect experiment',
            population=dict(frames=584,streams=8,objects=8),macro_add_drop_pp=drop,objects=objects,
            top3_fraction_of_drop=sum(x['fraction_of_total_drop'] for x in objects if x['object_id'] in (1,4,8)),
            gelatin_late_segment={s:dict(frames=len(rr),rotation_deg=float(np.mean([r['rotation_deg'] for r in rr])),
                                        center_mm=float(np.mean([r['center_mm'] for r in rr])),
                                        adds_success=float(np.mean([r['adds_01'] for r in rr]))) for s,rr in late.items()},
            predictions_sha256={s:hashlib.sha256((run/f'val_{s}/predictions.jsonl').read_bytes()).hexdigest() for s in rows})
(out/'val_22000_24000.json').write_text(json.dumps(result,indent=2))
fig,axes=plt.subplots(3,1,figsize=(8,6.5),sharex=True,layout='constrained')
for s,color in [(22000,'#167b9d'),(24000,'#db6233')]:
    rr=gelatin[s];x=[r['frame_index'] for r in rr]
    for ax,key in zip(axes[:2],('rotation_deg','center_mm')):
        ax.plot(x,[r[key] for r in rr],label=f'{s//1000}k',color=color,lw=2)
axes[0].set_ylabel('Rotation error (deg)');axes[0].legend(loc='upper left',ncol=2)
axes[0].axhline(90,color='gray',ls=':',lw=1)
axes[0].set_title('Gelatin box: orientation locks near 90 degrees after occlusion')
axes[1].set_ylabel('Center error (mm)')
axes[2].plot(x,[r['visibility'] for r in gelatin[24000]],color='#46515d',lw=1.8)
axes[2].set_ylabel('Visibility proxy');axes[2].set_xlabel('Frame index');axes[2].set_ylim(0,1.05)
for ax in axes:
    ax.axvspan(28,41,color='gray',alpha=.12)
    ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False)
fig.savefig(out/'gelatin_rotation_regression.png',dpi=170)
plt.close(fig)
print(json.dumps({k:v for k,v in result.items() if k!='objects'},indent=2))
