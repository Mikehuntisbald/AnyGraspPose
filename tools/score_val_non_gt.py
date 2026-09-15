"""Score sealed non-GT val predictions; annotations are accessed only in this stage."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.config import check_data_gate
from lip.engine.stream_checkpoint import sha
from lip.geometry.so3 import center_pose
from lip.evaluation.metrics import errors
from val_non_gt_common import val_streams


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['object_id']].append(row)
    return dict(frames=len(rows),objects=len(groups),missing_pose_frames=sum(r['pose_centered'] is None for r in rows),
        **{metric:100*float(np.mean([np.mean([r[metric] for r in group]) for group in groups.values()])) if groups else None
           for metric in ('add_01','adds_01','adds_005')})


def verify_method_policy(manifest,allow_foundationpose_baseline=False):
    assert manifest['gt_pose_reads']==manifest['gt_mask_reads']==manifest['hand_annotation_reads']==0
    if allow_foundationpose_baseline:
        assert manifest['architecture_id']=='foundationpose_tracking_v1'
        assert manifest['fp_calls']==manifest['frames']-manifest['missing_pose_frames']-manifest['initialized_streams']
        assert manifest['fp_iterations']==2 and manifest['fp_initialization_calls']==0
        assert manifest['access_audit']['foundationpose_reads_authorized']
    else:assert manifest['fp_calls']==0


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('run','index-root','visibility-reference','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--world',type=int,required=True);p.add_argument('--allow-subset',action='store_true')
    p.add_argument('--allow-foundationpose-baseline',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);audit=check_data_gate(a.index_root);manifests=[];rows=[]
    for rank in range(a.world):
        folder=a.run/f'rank{rank}';m=json.loads((folder/'manifest.json').read_text());assert m['completed']
        verify_method_policy(m,a.allow_foundationpose_baseline)
        assert m['initialization_uses_gt_pose'] is False and (not m['subset'] or a.allow_subset)
        assert sha(folder/'predictions.jsonl')==m['predictions_sha256']
        counts=m['access_audit']['counts'];assert not any(v for k,v in counts.items() if k.startswith('denied_'))
        assert counts['validated_native_image_reads']==2*(m['frames']-m['missing_pose_frames'])
        rows.extend(map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()));manifests.append(m)
    for field in ('checkpoint_sha256','config_sha256','source_sha256','split_hash','mesh_hash','initializers_sha256','expected_streams','expected_frames'):
        assert all(m[field]==manifests[0][field] for m in manifests),field
    m=manifests[0];assert m['split']=='val' and all(m[k]==audit[k] for k in ('split_hash','mesh_hash'))
    keys={(r['stream_id'],r['frame_index']) for r in rows};assert len(keys)==len(rows)==m['expected_frames']
    actual=sorted({r['stream_id'] for r in rows});assert actual==m['expected_streams']
    if not a.allow_subset:assert len(rows)==23200 and len(actual)==320
    reference=json.loads((a.visibility_reference/'manifest.json').read_text())
    assert reference['completed'] and reference['population_verified'] and reference['split']=='val'
    assert all(reference[k]==audit[k] for k in ('split_hash','mesh_hash'))
    vis={(r['stream_id'],r['frame_index']):r for r in map(json.loads,(a.visibility_reference/'predictions.jsonl').read_text().splitlines())}
    assert keys<=set(vis)
    registry={s['stream_id']:s for s in val_streams(a.index_root)};by_stream=defaultdict(list)
    for row in rows:by_stream[row['stream_id']].append(row)
    a.out.mkdir(parents=True,exist_ok=False);scored=[];initial_quality={}
    for sid,group in sorted(by_stream.items()):
        stream=registry[sid]
        with np.load(a.index_root/stream['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
        with np.load(a.index_root/stream['pose_cache']) as z:frames=z['frames'].copy();poses=z['poses'].copy()
        assert np.array_equal(frames,np.arange(stream['num_frames']))
        gt=center_pose(torch.from_numpy(poses),torch.from_numpy(mesh['center']))
        init=None
        for row in sorted(group,key=lambda r:r['frame_index']):
            frame=row['frame_index'];annotation=vis[sid,frame]
            assert row['object_id']==annotation['object_id']==stream['object_id']
            metric=errors(torch.tensor(row['pose_centered'],dtype=torch.float32),gt[frame].float(),mesh['vertices'],float(mesh['diameter']),dtype='f4') if row['pose_centered'] is not None else dict(
                add_01=False,adds_01=False,adds_005=False,add_m=None,adds_m=None,rotation_deg=None,center_mm=None)
            row=dict(row,**metric,visibility=annotation['visibility'],moving=annotation['moving'],
                diameter_m=float(mesh['diameter']),physical_sequence='/'.join(sid.split('/')[:2]))
            if row['initialization']:
                assert init is None;init=dict(frame=frame,good=bool(row['adds_005']),very_bad=not row['adds_01'])
            row['updates_after_initialization']=frame-init['frame'] if init is not None else None
            row['initial_bad']=not init['good'] if init is not None else None
            scored.append(row)
        initial_quality[sid]=init
    populations=dict(all=scored,excluding_initialization=[r for r in scored if not r['initialization']],
        visibility_lt_05=[r for r in scored if r['visibility'] is not None and r['visibility']<.5],
        visibility_lt_03=[r for r in scored if r['visibility'] is not None and r['visibility']<.3],
        after_initialization=[r for r in scored if r['updates_after_initialization'] is not None and r['updates_after_initialization']>0],
        bad_initial_first8=[r for r in scored if r['initial_bad'] and 0<r['updates_after_initialization']<=8])
    report=dict(completed=True,checkpoint_sha256=m['checkpoint_sha256'],initializers_sha256=m['initializers_sha256'],
        frames=len(scored),streams=len(actual),initial_quality=initial_quality,populations={n:summarize(rs) for n,rs in populations.items()},
        scope='Native full s0 val with real non-GT initialization. Object-macro ADD thresholds, not official test BOP AR. Primary all-frame scores include initialization and every missing-pose failure. Continuous errors are conditional on an emitted pose.',
        visibility_definition='Reused fixed native-val segmentation/silhouette visibility after inference; not official BOP visib_fract.',
        visibility_reference_sha256=sha(a.visibility_reference/'predictions.jsonl'))
    raw=''.join(json.dumps(row,allow_nan=False)+'\n' for row in scored);(a.out/'predictions.jsonl').write_text(raw)
    (a.out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    combined_audit=dict(m['access_audit'],counts={key:sum(x['access_audit']['counts'].get(key,0) for x in manifests)
        for key in set().union(*(x['access_audit']['counts'] for x in manifests))},
        merged_inference_shards=a.world)
    final=dict(m,completed=True,population_verified=True,frames=len(scored),streams=actual,world=a.world,rank=None,
        extra_inner_attempts=sum(x.get('extra_inner_attempts',0) for x in manifests),inner_fallbacks=sum(x.get('inner_fallbacks',0) for x in manifests),
        fp_calls=sum(x['fp_calls'] for x in manifests),missing_pose_frames=sum(x['missing_pose_frames'] for x in manifests),
        initialized_streams=sum(x['initialized_streams'] for x in manifests),
        access_audit=combined_audit,
        predictions_sha256=sha(a.out/'predictions.jsonl'),inference_shards=[dict(rank=x['rank'],predictions_sha256=x['predictions_sha256']) for x in manifests],
        scoring_entrypoint_sha256=sha(Path(__file__)),gt_access_stage='Scoring only, after all prediction hashes verified')
    (a.out/'manifest.json').write_text(json.dumps(final,indent=2));print(json.dumps(report['populations'],indent=2))


if __name__=='__main__':main()
