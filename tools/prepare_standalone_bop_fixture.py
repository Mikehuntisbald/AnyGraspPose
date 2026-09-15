"""Make a clearly labeled BOP-interface fixture using train RGB-D and a synthetic pose."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from lip.benchmark.bop_io import csv_row,FIELDS


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('source-data-root','index-root','fp-root','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',action='append',required=True,help='NAME=checkpoint')
    p.add_argument('--frames',type=int,default=16)
    a=p.parse_args();assert a.frames>=12
    root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=False)
    streams=sorted([r for r in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())
                    if r['split']=='train' and r['num_frames']>=a.frames],key=lambda r:r['stream_id'])
    s=streams[0];audit=json.loads((a.index_root/'audit.json').read_text())
    raw=a.out/'data';scene=raw/'bop/s0/test/000001'
    for name in ('rgb','depth'): (scene/name).mkdir(parents=True)
    files=[]
    for frame in range(a.frames):
        for source_name,destination in ((f'color_{frame:06d}.jpg',scene/'rgb'/f'{frame:06d}.jpg'),
                                         (f'aligned_depth_to_color_{frame:06d}.png',scene/'depth'/f'{frame:06d}.png')):
            source=a.source_data_root/s['relative_dir']/source_name
            shutil.copy2(source,destination);assert sha(source)==sha(destination)
            files.append(dict(source=str(source),destination=str(destination),sha256=sha(source)))
    camera={str(frame):dict(cam_K=np.asarray(s['intrinsics']).reshape(-1).tolist(),
                            depth_scale=1000*float(audit['depth_scale_to_m'])) for frame in range(a.frames)}
    (scene/'scene_camera.json').write_text(json.dumps(camera))
    mesh_source=a.source_data_root/s['mesh_path'];mesh_destination=raw/s['mesh_path']
    mesh_destination.parent.mkdir(parents=True);shutil.copy2(mesh_source,mesh_destination)
    # Canary files are synthetic invalid contents, not copied annotations.
    (scene/'scene_gt.json').write_text('FORBIDDEN SYNTHETIC ANNOTATION CANARY')
    (scene/'mask_visib').mkdir();(scene/'mask_visib/000002_000000.png').write_text('FORBIDDEN')
    (raw/'labels_000002.npz').write_text('FORBIDDEN')
    targets=[dict(scene_id=1,im_id=f,obj_id=s['object_id'],inst_count=1) for f in (0,2,8,a.frames-1)]
    targets += [dict(scene_id=2,im_id=f,obj_id=s['object_id'],inst_count=1) for f in (0,a.frames-1)]
    target_path=a.out/'targets.json';target_path.write_text(json.dumps(targets,indent=2))
    initial=np.eye(4);angle=.2;initial[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    initial[:3,3]=[.01,-.01,.65]
    bad=initial.copy();bad[:3,:3]=np.diag([-1.,1.,1.])
    low=initial.copy();low[2,3]=.55
    future=np.eye(4);future[:3,3]=[-.05,.05,.95]
    initializer_path=a.out/'fixture_initializer.csv'
    with initializer_path.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader()
        for frame,score,pose in ((0,1.,bad),(2,.2,low),(2,.8,initial),(8,.99,future)):
            writer.writerow(csv_row(dict(scene_id=1,im_id=frame,obj_id=s['object_id'],score=score),pose))
    configurations={}
    for value in a.checkpoint:
        name,path=value.split('=',1);checkpoint=Path(path)
        if not name or name in configurations or '/' in name:raise ValueError('Distinct simple names required')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        config=dict(state['config'],precision='fp32',cpu_threads=2)
        folder=a.out/name;folder.mkdir();config_path=folder/'config.yaml'
        config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        protocol=dict(frozen=True,fixture_only=True,initializer_source='Synthetic legal pose; interface test, not PoseCNN output',
            history='causal_full_frames',fp_calls=0,source_sha256=source_hash(),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            checkpoint_training_source_sha256=state.get('source_sha256'),
            config=str(config_path),config_sha256=sha(config_path),initializer_csv=str(initializer_path),initializer_sha256=sha(initializer_path),
            targets=str(target_path),targets_sha256=sha(target_path),data_root=str(raw),index_root=str(a.index_root),fp_root=str(a.fp_root),
            mesh_cache=str(a.out/'mesh_cache'/name),fps=float(audit['fps']),methods=['lip_temporal','lip_no_feature_history'],
            tools_sha256={n:sha(root/'tools'/n) for n in ('infer_lip_tracking_bop.py','streaming_bop_utils.py','standalone_bop_state.py')})
        protocol_path=folder/'protocol.json';protocol_path.write_text(json.dumps(protocol,indent=2))
        configurations[name]=dict(protocol=str(protocol_path),protocol_sha256=sha(protocol_path),checkpoint_sha256=sha(checkpoint))
    receipt=dict(completed=True,fixture_only=True,native_split='train',source_stream=s['stream_id'],object_id=s['object_id'],
        image_files=files,mesh_source=str(mesh_source),mesh_sha256=sha(mesh_source),configurations=configurations,
        synthetic_initial_pose=initial.tolist(),synthetic_future_pose=future.tolist(),
        expected_target_keys=[[1,f,s['object_id']] for f in (2,8,a.frames-1)],expected_processed_object_frames=a.frames-2,
        missing_target_instances=3,
        scope='Only train RGB-D, calibration metadata and an object mesh are copied. No source pose cache or label is read to construct the initializer. BOP test directory naming is an interface fixture, not s0 test data. Tests earliest legal initializer, confidence ordering, missing initialization, full intervening frames, cache boundaries and original-pose millimeter export. No accuracy metric or evaluator is run.')
    (a.out/'fixture.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='image_files'},indent=2))


if __name__=='__main__':main()
