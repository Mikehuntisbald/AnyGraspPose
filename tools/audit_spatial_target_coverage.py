"""Attribute missing train correspondences to fixed visibility/depth predicates."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.data.stream_clips import StreamClips
from lip.engine.stream_features import build_current_features
from lip.geometry.renderer import Renderer
from lip.losses_spatial_alignment import sample_nearest
from lip.engine.stream_checkpoint import sha


def main():
    root=Path(__file__).resolve().parents[1];e=json.loads((root/'runs/factorial/experiment.json').read_text());out=root/'runs/target_audit/coverage.json';assert not out.exists()
    old=json.loads(Path('/mnt/why/dexycb_lip/alignment_diagnosis_20260915/runs/train_diagnostic/spec.json').read_text())['cohort']
    data=StreamClips(e['data_root'],e['index_root'],0,1,fixed=json.loads(Path(e['training_manifest']).read_text()),decode_threads=0,
        external_initializers=e['train_initializers'],external_initializers_sha256=e['train_initializers_sha256'],real_initialization_probability=.5,include_initial_observation=True)
    torch.set_num_threads(2);renderer=Renderer('cpu');rows=[]
    axis=(torch.arange(14).float()+.5)*16-.5;yy,xx=torch.meshgrid(axis,axis,indexing='ij');uv0=torch.stack((xx,yy),-1).reshape(1,196,2)
    for item in old['fit']+old['probe']:
        s=data[item['manifest_index']];rgb=s['rgb'][0].float()/255;depth=s['depth'][0].float()*s['depth_scale'];d=float(s['mesh']['diameter'])
        f,diag=build_current_features(rgb,depth,s['real_initial_pose'],s['k'],s['mesh'],renderer,s['timestamps'][1],s['timestamps'][0],size=224)
        target=s['targets'][0];gt_depth,_=renderer(s['mesh'],target,diag['K_crop'],224)
        source=sample_nearest(f['geometry'][None,3:7],uv0)[0];xyz=source[:,1:]*d
        camera=xyz@target[:3,:3].T+target[:3,3];z=camera[:,2];project=camera@diag['K_crop'].T;uv=project[:,:2]/project[:,2:].clamp_min(1e-6)
        gd=sample_nearest(gt_depth[None],uv[None])[0,:,0]
        raw=torch.cat((uv,torch.ones(196,1)),-1)@torch.linalg.inv(diag['A']).T;raw=raw[:,:2]/raw[:,2:]
        od=sample_nearest(depth[None],raw[None])[0,:,0]
        source_valid=source[:,0]>.5;inside=source_valid&(z>.001)&(uv>=0).all(-1)&(uv<=223).all(-1)
        self_visible=inside&(gd>0)&((z-gd).abs()<=max(.002,.01*d))
        observed=self_visible&(od>0);consistent=observed&((z-od).abs()<=max(.005,.02*d))
        def quantile(v):return torch.quantile(v,torch.tensor([0.,.25,.5,.75,1.])).tolist() if len(v) else None
        rows.append(dict(object_id=s['stream']['object_id'],stream_id=s['stream']['stream_id'],manifest_index=item['manifest_index'],
            symmetric=s['stream']['object_id'] in e['symmetric_object_ids'],cad_keys=int(source_valid.sum()),projected_inside=int(inside.sum()),
            self_visible=int(self_visible.sum()),observed_depth_valid=int(observed.sum()),depth_consistent=int(consistent.sum()),
            observed_minus_gt_mm_quantiles=quantile((od[observed]-z[observed])*1000),depth_tolerance_mm=max(.005,.02*d)*1000,
            gt_self_error_mm_quantiles=quantile((gd[inside]-z[inside]).abs()*1000)))
    result=dict(completed=True,split='train',frames=16,scope='One unaugmented first-update frame per pre-fixed diagnostic clip; no optimization, tolerance change or dataset edits. CPU renderer diagnostic.',
        entrypoint_sha256=sha(Path(__file__)),rows=rows)
    out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
