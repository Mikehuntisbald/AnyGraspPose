"""Bounded train-only augmentation coverage and real closed-loop checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.data.stream_clips import StreamClips
from lip.data.perturb import noisy_history
from lip.data.temporal_occlusion import make_plan,composite
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_training import StreamTrainingModule
from lip.engine.stream_features import mesh_to_device
from lip.engine.stream_checkpoint import source_hash,sha
from lip.geometry.renderer import Renderer


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args()
    r=a.experiment;e=json.loads((r/'experiment.json').read_text());torch.set_num_threads(2)
    c=load_stream_config(e['arms']['control']['config']);renderer=Renderer('cuda');burn=c['burn_in_frames'];unroll=c['supervised_unroll_frames'];total=burn+unroll;startup=c.get('temporal_occlusion_startup_probability',0.)
    ds=StreamClips(e['data_root'],e['index_root'],burn,unroll,seed=42,fixed=json.loads(Path(e['training_manifest']).read_text()))
    model=make_model(c).cuda().eval();model.load_state_dict(torch.load(e['arms']['control']['init'],map_location='cpu',weights_only=False)['model'])
    samples=[ds[i] for i in range(2)];fingerprints=[hashlib.sha256(s['rgb'].numpy().tobytes()+s['depth'].numpy().tobytes()).hexdigest() for s in samples]
    absent=dict(c);absent.pop('temporal_occlusion_probability',None);absent.pop('temporal_occlusion_startup_probability',None)
    with torch.no_grad():
        baseline=StreamTrainingModule(model,absent,renderer)(samples,True)
        disabled=StreamTrainingModule(model,dict(c,temporal_occlusion_probability=0.,temporal_occlusion_startup_probability=0.),renderer)(samples,True)
        augmented=StreamTrainingModule(model,dict(c,temporal_occlusion_probability=1.),renderer)(samples,True)
        repeated=StreamTrainingModule(model,dict(c,temporal_occlusion_probability=1.),renderer)(samples,True)
    assert torch.equal(baseline['predictions'],disabled['predictions'])
    assert torch.equal(augmented['predictions'],repeated['predictions'])
    assert torch.isfinite(augmented['predictions']).all() and torch.isfinite(augmented['loss'])
    protected=0 if startup else burn+2
    assert torch.equal(baseline['predictions'][:,:protected],augmented['predictions'][:,:protected])
    assert not torch.equal(baseline['predictions'],augmented['predictions'])
    assert fingerprints==[hashlib.sha256(s['rgb'].numpy().tobytes()+s['depth'].numpy().tobytes()).hexdigest() for s in samples]
    rotation=augmented['predictions'][...,:3,:3];orth=(rotation.transpose(-1,-2)@rotation-torch.eye(3,device='cuda')).abs().max()
    assert float(orth)<1e-3 and (augmented['predictions'][...,2,3]>.01).all()
    records=[];figures=[]
    # Current GT is used only to score already-composited images, never to
    # construct an occluder or to reposition it toward the later object.
    for idx in range(32):
        s=samples[idx] if idx<2 else ds[idx];mesh=mesh_to_device(s['mesh'],'cuda');initial=s['initial_pose'].cuda().float();k=s['k'].cuda().float()
        if c['initial_pose_noise']:initial=noisy_history(initial[None],mesh['diameter'],torch.Generator().manual_seed(s['sample']['seed']))[0][0]
        first=s['rgb'][0].cuda().float()/255
        plan=make_plan(first,initial,k,mesh['vertices'],mesh['diameter'],s['sample']['seed'],total,burn,1.,startup)
        assert plan is not None and (plan.start==0 or plan.start>=burn+2) and plan.end<=total-4
        i=(plan.start+plan.end)//2;rgb=s['rgb'][i].cuda().float()/255;depth=s['depth'][i].cuda().float()*s['depth_scale']
        aug,dep,mask=composite(rgb,depth,plan,i)
        assert torch.isfinite(aug).all() and torch.isfinite(dep).all()
        h,w=rgb.shape[-2:];rendered,_=renderer(mesh,s['targets'][i].cuda(),k,max(h,w));rendered=rendered[:,:h,:w]
        silhouette=rendered>0;visible=silhouette&(depth>0)&((depth-rendered).abs()<(.003+.01*mesh['diameter']))
        coverage=float((mask&silhouette).sum()/silhouette.sum().clamp_min(1))
        coverage_visible=float((mask&visible).sum()/visible.sum().clamp_min(1))
        record=dict(sample_index=idx,sample=s['sample'],stream_id=s['stream']['stream_id'],frame=int(s['frames'][i+1]),
            start=plan.start,end=plan.end,frame_index_in_fragment=i,mask_pixel_fraction=float(mask.float().mean()),
            rendered_silhouette_pixels=int(silhouette.sum()),depth_consistent_pixels=int(visible.sum()),
            covered_silhouette_fraction=coverage,covered_depth_consistent_fraction=coverage_visible,plane_depth_m=plan.depth_m)
        records.append(record)
        if idx<4:
            original=Image.fromarray((rgb.permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            after=Image.fromarray((aug.permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            canvas=Image.new('RGB',(2*w,h+32));canvas.paste(original,(0,32));canvas.paste(after,(w,32))
            ImageDraw.Draw(canvas).text((8,8),f'Train example {idx}: original | synthetic RGB-D occlusion; silhouette coverage {coverage:.1%}',fill='white')
            path=r/f'occlusion_example_{idx}.png';canvas.save(path);figures.append(dict(file=path.name,sha256=sha(path)))
    hit=sum(x['covered_silhouette_fraction']>.3 for x in records)
    # This is a bounded effectiveness gate on train examples, not a val gain.
    assert hit>=8,'Fewer than 8 of 32 planned occluders cover >30% of projected object'
    report=dict(passed=True,scope='32 train fragments, probability forced to 1 for coverage; midpoint only. No val/test or hand labels. GT rendering is diagnostic after compositing.',
        source_sha256=source_hash(),checkpoint_sha256=sha(e['arms']['control']['init']),training_manifest_sha256=sha(e['training_manifest']),
        no_augmentation_missing_vs_zero_bitwise=True,augmented_repeat_bitwise=True,guaranteed_unaugmented_prefix=protected,first_ten_frames_unchanged=bool(torch.equal(baseline['predictions'][:,:10],augmented['predictions'][:,:10])),raw_tensors_unchanged=True,
        startup_probability=startup,startup_plans=sum(row['start']==0 for row in records),
        legal_pose=True,max_rotation_orthogonality_error=float(orth),real_closed_loop_frames=2*total,
        baseline_loss=float(baseline['loss']),augmented_loss=float(augmented['loss']),
        augmented_pixel_fraction=float(augmented['occlusion_diagnostics'][0]),augmented_active_frame_fraction=float(augmented['occlusion_diagnostics'][1]),
        coverage_over_30_percent_clips=hit,clips=32,records=records,figures=figures,visual_review_completed=False)
    (r/'occlusion_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in report.items() if k not in ('records','figures')},indent=2))


if __name__=='__main__':main()
