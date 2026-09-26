"""Teacher-only representability diagnostic, without loading a learned model.

GT selects target surfaces and hypothetical reference views; nothing here is a
deployable prediction. Distances to finite raster samples are NOT lower bounds
for bilinear interpolation (interpolated points can lie between samples).
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_geometry_supervision import AppearanceOnlyStore
from lip.unified.training import Factory
from lip.unified.features import prepare_scene
from lip.unified.cad import surface_points
from lip.geometry.so3 import exp


def distances(query,reference):
    if not len(reference):return torch.full((len(query),),float('inf'),device=query.device)
    return torch.cat([torch.cdist(q[None],reference[None],compute_mode='donot_use_mm_for_euclid_dist')[0].amin(1) for q in query.split(128)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    c=yaml.safe_load(Path(a.config).read_text());factory=Factory(c,SimpleNamespace(encoder=None),AppearanceOnlyStore())
    a.out.mkdir(parents=True,exist_ok=False);rows=[]
    records=json.loads((a.audit/'audit.json').read_text())['cases']
    with torch.no_grad(),(a.out/'frames.jsonl').open('w') as log:
        for i,record in enumerate(records):
            e,(truth,masks)=factory.sample(record['seed'],frames=1)
            assert e.stream==record['stream']
            s=prepare_scene(e.rgb[0],e.depth[0],e.initial,e.mesh,e.k,e.times[0],e.stream,e.cad,factory.renderer,fast=True)
            target=np.load(a.audit/f'case{i:02d}.npz')
            full=surface_points(s.cad['appearance'],count=8192)['coord']/s.diameter
            query={}
            for kind in ('real','proxy'):
                xyz=torch.tensor(target['cad_xyz'].transpose(1,2,0)[target[kind]],device='cuda')
                if len(xyz)>512:xyz=xyz[torch.linspace(0,len(xyz)-1,512,device='cuda').long()]
                if len(xyz):query[kind]=xyz
            complete={k:distances(q,full) for k,q in query.items()}
            for angle in (0,10,45,90,180):
                pose=truth[0].clone();rotation=torch.zeros(3,device='cuda');rotation[i%3]=angle*torch.pi/180
                pose[:3,:3]=exp(rotation)@pose[:3,:3]
                render=factory.renderer(s.cad['appearance'],pose,s.k_crop,224)
                reference=render['xyz'].permute(1,2,0)[render['mask']]/s.diameter
                for kind,q in query.items():
                    error=distances(q,reference);f=complete[kind]
                    if angle==0:assert float(error.max()*s.diameter*1000)<.02
                    row=dict(seed=record['seed'],stream=e.stream,kind=kind,reference_rotation_deg=angle,queries=len(q),
                             raster_points=len(reference),full_surface_points=len(full),
                             raster_nearest_mm=float(error.mean()*s.diameter*1000),full_surface_nearest_mm=float(f.mean()*s.diameter*1000),
                             raster_within_3pct_d=float((error<.03).float().mean()),full_surface_within_3pct_d=float((f<.03).float().mean()))
                    rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
    import statistics
    table={}
    for angle in (0,10,45,90,180):
        table[str(angle)]={}
        for kind in ('real','proxy'):
            selected=[r for r in rows if r['reference_rotation_deg']==angle and r['kind']==kind]
            by_sequence={}
            for r in selected:
                physical='/'.join(r['stream'].split('|')[0].split('/')[:2]);by_sequence.setdefault(physical,[]).append(r)
            table[str(angle)][kind]={k:statistics.mean(statistics.mean(x[k] for x in seq) for seq in by_sequence.values()) for k in ('raster_nearest_mm','full_surface_nearest_mm','raster_within_3pct_d','full_surface_within_3pct_d')}
    result=dict(completed=True,training=False,model_loaded=False,frames=len(records),table=table,
                scope='Teacher-only surface availability at controlled view rotations; not learned correspondence or deployable pose. Finite raster nearest distance is not a bilinear-lookup lower bound.',
                reduction='Equal physical-sequence mass; up to512 deterministic target pixels per region/frame')
    (a.out/'outcome.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
