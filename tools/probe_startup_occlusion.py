"""CPU-only real-image coverage and archived-default compatibility for startup occlusion."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.data.stream_clips import StreamClips
from lip.data.index import read_frame
from lip.data.perturb import noisy_history
from lip.data.temporal_occlusion import make_plan,composite
from lip.geometry.so3 import center_pose
from lip.geometry.renderer import Renderer
from lip.engine.stream_checkpoint import sha,source_hash


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('previous-runtime','manifest','data-root','index-root','out'):
        p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=False)
    old_path=a.previous_runtime/'src/lip/data/temporal_occlusion.py'
    spec=importlib.util.spec_from_file_location('archived_occlusion',old_path);old=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=old;spec.loader.exec_module(old)
    dataset=StreamClips(a.data_root,a.index_root,8,48,fixed=json.loads(a.manifest.read_text()))
    renderer=Renderer('cpu');records=[];figures=[];started=time.time()
    for i in range(32):
        item=dataset.choose(i);s=dataset.streams[item['stream']];poses=dataset.poses[item['stream']]
        mesh=dataset.meshes[s['mesh_cache']];start=item['start'];k=torch.tensor(s['intrinsics'],dtype=torch.float32)
        initial=center_pose(torch.from_numpy(poses['poses'][start].copy()),torch.from_numpy(mesh['center']))
        initial=noisy_history(initial[None],float(mesh['diameter']),torch.Generator().manual_seed(item['seed']))[0][0]
        frame0=int(poses['frames'][start+1]);rgb0,depth0=read_frame(a.data_root,s,frame0,dataset.audit['depth_scale_to_m'])
        rgb0=torch.from_numpy(rgb0).float();depth0=torch.from_numpy(depth0).float()
        args=(rgb0,initial,k,torch.from_numpy(mesh['vertices']),float(mesh['diameter']),item['seed'],56,8,1.)
        archived=old.make_plan(*args);regular=make_plan(*args);startup=make_plan(*args,startup_probability=1.)
        assert archived is not None and regular is not None and startup.start==0
        for field in archived.__dataclass_fields__:
            aa,bb=getattr(archived,field),getattr(regular,field)
            assert torch.equal(aa,bb) if isinstance(aa,torch.Tensor) else aa==bb
        assert startup.end==regular.end-regular.start and startup.end<=52
        row=dict(sample_index=i,sample=item,stream_id=s['stream_id'],object_id=s['object_id'],
            default_start=regular.start,default_end=regular.end,startup_start=startup.start,startup_end=startup.end,checks=[])
        for index in (0,(startup.start+startup.end)//2):
            frame=int(poses['frames'][start+1+index])
            if index==0:rgb,depth=rgb0,depth0
            else:
                rr,dd=read_frame(a.data_root,s,frame,dataset.audit['depth_scale_to_m']);rgb=torch.from_numpy(rr).float();depth=torch.from_numpy(dd).float()
            before_rgb=rgb.clone();before_depth=depth.clone();aug,dep,mask=composite(rgb,depth,startup,index)
            assert torch.equal(rgb,before_rgb) and torch.equal(depth,before_depth)
            assert torch.isfinite(aug).all() and torch.isfinite(dep).all()
            # GT enters only after a complete augmented observation exists.
            target=center_pose(torch.from_numpy(poses['poses'][start+1+index].copy()),torch.from_numpy(mesh['center']))
            h,w=rgb.shape[-2:];rd,_=renderer(mesh,target,k,max(h,w));rd=rd[:,:h,:w]
            silhouette=rd>0;visible=silhouette&(depth>0)&((depth-rd).abs()<(.003+.01*float(mesh['diameter'])))
            coverage=float((mask&silhouette).sum()/silhouette.sum().clamp_min(1))
            row['checks'].append(dict(observation_index=index,frame=frame,projected_pixels=int(silhouette.sum()),
                depth_consistent_pixels=int(visible.sum()),covered_silhouette_fraction=coverage,
                covered_depth_consistent_fraction=float((mask&visible).sum()/visible.sum().clamp_min(1)),mask_pixel_fraction=float(mask.float().mean())))
            if i<4 and index==0:
                original=Image.fromarray((rgb.permute(1,2,0).numpy()*255).clip(0,255).astype('uint8'))
                modified=Image.fromarray((aug.permute(1,2,0).numpy()*255).clip(0,255).astype('uint8'))
                canvas=Image.new('RGB',(w*2,h+32));canvas.paste(original,(0,32));canvas.paste(modified,(w,32))
                ImageDraw.Draw(canvas).text((8,8),f'Train startup {i}; original | synthetic; projected coverage {coverage:.1%}',fill='white')
                path=a.out/f'startup_{i:02d}.png';canvas.save(path);figures.append(dict(file=path.name,sha256=sha(path)))
        records.append(row)
        with (a.out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(samples=i+1,seconds=time.time()-started))+'\n')
    hits=[sum(r['checks'][j]['covered_silhouette_fraction']>.3 for r in records) for j in (0,1)]
    report=dict(completed=True,passed=min(hits)>=8,scope='32 actual train fragments; CPU GT rendering only after compositing. Force occluder and startup probabilities to 1 to test coverage. No actor training, GPU usage, hand labels or validation accuracy claim.',
        source_sha256=source_hash(),script_sha256=sha(__file__),previous_generator_sha256=sha(old_path),manifest_sha256=sha(a.manifest),
        archived_default_plan_bitwise_equal=True,raw_tensors_unchanged=True,first_update_and_midpoint_hits_over30percent=hits,
        minimum_required_hits=8,records=records,figures=figures,visual_review_completed=False,human_verified=False)
    (a.out/'receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('records','figures')},indent=2))
    assert report['passed'],'Fewer than 8/32 startup plans cover 30% of projected target'


if __name__=='__main__':main()
