"""CPU-only CAD geometric coverage audit on frozen training-development cache."""
import argparse,json,sys,statistics
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);rows=[]
    for file in sorted(a.cache.glob('rank*/decoder_cache.pt')):
        for r in torch.load(file,weights_only=False):
            available=r['cad_available'][0];bank=r['cad_geometry'][0,available,:3].float()
            if not len(bank):continue
            row=dict(seed=r['seed'],stream=r['stream'])
            for name,mask,xyz in [('real',r['real_mask'],r['xyz']),('proxy',r['proxy_mask'],r['xyz']),('visible',r['observed_mask'],r['observed_xyz'])]:
                points=xyz[0].permute(1,2,0)[mask[0,0]].float()
                if len(points)>2048:points=points[torch.linspace(0,len(points)-1,2048).long()]
                if not len(points):row[name]=None;continue
                distance=torch.cdist(points,bank).min(-1).values*r['diameter']*1000
                row[name]=dict(points=len(points),nearest_anchor_mm=float(distance.mean()),median_mm=float(distance.median()),p90_mm=float(distance.quantile(.9)),fraction_gt20mm=float((distance>20).float().mean()))
            rows.append(row)
    result=dict(scope='Training-split cache only; geometric quantization, not learned matching or native pose',records=len(rows),max_points_per_region=2048,regions={},rows=rows)
    for name in ('real','proxy','visible'):
        selected=[r[name] for r in rows if r[name] is not None]
        result['regions'][name]=dict(nonempty_records=len(selected),mean_frame_nearest_anchor_mm=statistics.mean(x['nearest_anchor_mm'] for x in selected),mean_frame_fraction_gt20mm=statistics.mean(x['fraction_gt20mm'] for x in selected))
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['regions'],indent=2))

if __name__=='__main__':main()
