"""Align validation diagnostics with the critic's sampled ADD-S label definition."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',required=True,type=Path);p.add_argument('--index-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())};groups={};rows=[]
    for r in map(json.loads,(a.evaluation/'predictions.jsonl').read_text().splitlines()):groups.setdefault(r['stream_id'],[]).append(r)
    with torch.no_grad():
        for sid,rs in groups.items():
            rs.sort(key=lambda r:r['frame_index']);s=streams[sid]
            with np.load(a.index_root/s['mesh_cache']) as z:points=torch.tensor(z['points'],device='cuda');center=torch.tensor(z['center'],device='cuda');diameter=float(z['diameter'])
            with np.load(a.index_root/s['pose_cache']) as z:gt={int(f):t for f,t in zip(z['frames'],z['poses'])}
            for start in range(1,len(rs),32):
                part=rs[start:start+32];before=torch.tensor([r['pose_centered'] for r in rs[start-1:start+len(part)-1]],device='cuda');after=torch.tensor([r['pose_centered'] for r in part],device='cuda')
                target=torch.tensor(np.stack([gt[r['frame_index']] for r in part]),device='cuda');target[:,:3,3]+=target[:,:3,:3]@center
                ground=points@target[:,:3,:3].transpose(-1,-2)+target[:,None,:3,3];errs=[]
                for pose in (before,after):
                    xyz=points@pose[:,:3,:3].transpose(-1,-2)+pose[:,None,:3,3]
                    errs.append(torch.cdist(xyz,ground,compute_mode='donot_use_mm_for_euclid_dist').min(-1).values.mean(-1)/diameter)
                errors=torch.stack(errs,1).cpu().numpy()
                for offset,(r,error) in enumerate(zip(part,errors)):
                    rows.append(dict(stream_id=sid,frame=r['frame_index'],startup=start+offset<=8,visibility=r['visibility'],support=r.get('observation_support'),
                        before_adds_d=float(error[0]),after_adds_d=float(error[1]),harmful=bool(error[1]>error[0]+.005),helpful=bool(error[1]<error[0]-.005)))
    pops={'all':rows,'first_8_updates':[r for r in rows if r['startup']],'after_first_8':[r for r in rows if not r['startup']],
        'visibility_lt_05':[r for r in rows if r['visibility'] is not None and r['visibility']<.5],
        'visibility_lt_03':[r for r in rows if r['visibility'] is not None and r['visibility']<.3]}
    report=dict(completed=True,scope='Single-step sampled ADD-S/d, fixed 512 surface points, margin .005d; matches quality-head label arithmetic; not a recursive gate ablation',populations={})
    for name,rs in pops.items():
        q=[r for r in rs if r['support'] is not None]
        report['populations'][name]=dict(frames=len(rs),harmful_fraction=float(np.mean([r['harmful'] for r in rs])),helpful_fraction=float(np.mean([r['helpful'] for r in rs])),
            mean_change_d=float(np.mean([r['after_adds_d']-r['before_adds_d'] for r in rs])),
            support_safe_auc=float(roc_auc_score([not r['harmful'] for r in q],[r['support'] for r in q])) if len(set(r['harmful'] for r in q))==2 else None)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2));a.out.with_suffix('.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
