"""Independent LIP tracking with direct PoseCNN initialization; zero FP imports or calls."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
from collections import Counter
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lip.benchmark.bop_io import load_predictions, csv_row, FIELDS
from lip.engine.stream_checkpoint import sha, source_hash, load_init
from lip.engine.stream_config import load_stream_config, make_model
from lip.engine.config import check_data_gate
from lip.geometry.mesh import mesh_metadata
from lip.geometry.renderer import Renderer
from lip.data.index import CLASSES
from streaming_bop_utils import plan_streams, expected_keys, clear_feature_history, hold_failed_step
from standalone_bop_state import InferenceAccessGuard,cache_summary,prime_initial_observation,standalone_methods


METHODS = ("lip_temporal", "lip_no_feature_history")

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=8)
    parser.add_argument('--smoke-streams', type=int)
    parser.add_argument('--smoke-frames', type=int)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    a = parser.parse_args()
    p = json.loads(a.protocol.read_text())
    methods = standalone_methods(p)
    assert p['frozen'] and p['history'] == 'causal_full_frames' and p['fp_calls'] == 0
    assert source_hash() == p['source_sha256']
    for field in ('checkpoint', 'initializer_csv', 'targets', 'config'):
        digest_field = 'initializer_sha256' if field == 'initializer_csv' else field+'_sha256'
        assert sha(p[field]) == p[digest_field], field
    execution = json.loads(a.execution.read_text()) if a.execution else None
    if execution:
        assert execution['protocol_sha256'] == sha(a.protocol)
        assert execution['workers'] == a.world
        assert sha(execution['assignments']) == execution['assignments_sha256']
    assert {'infer_lip_tracking_bop.py','streaming_bop_utils.py','standalone_bop_state.py'} <= set((execution or p)['tools_sha256'])
    for name, digest in (execution or p)['tools_sha256'].items():
        assert sha(Path(__file__).parent/name) == digest, name
    assert 0 <= a.rank < a.world
    raw_root = Path(p['data_root']).resolve()
    for destination in (a.out,Path(p['mesh_cache'])):
        if destination.resolve().is_relative_to(raw_root):raise ValueError('Inference outputs and mesh cache must be outside the raw data root')
    guard=InferenceAccessGuard(raw_root,p['index_root'],p['fp_root'])
    sys.addaudithook(guard)
    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    torch.manual_seed(42)
    np.random.seed(42)
    a.out.mkdir(parents=True, exist_ok=False)
    c = load_stream_config(p['config'])
    device=torch.device(a.device)
    if device.type=='cpu' and c['precision']!='fp32':raise ValueError('CPU interface checks require an explicit FP32 config')
    model = make_model(c).to(device).eval()
    load_init(p['checkpoint'], model, check_data_gate(p['index_root']), c)
    targets = json.loads(Path(p['targets']).read_text())
    plans = plan_streams(targets, load_predictions(p['initializer_csv']))
    if execution:
        identities = {tuple(x) for x in json.loads(Path(execution['assignments']).read_text())[str(a.rank)]}
        plans = [x for x in plans if (x['scene_id'], x['obj_id']) in identities]
        assert len(plans) == len(identities)
    else:
        plans = [x for x in plans if x['scene_id'] % a.world == a.rank]
    plans.sort(key=lambda x: (x['obj_id'], x['scene_id']))
    if a.smoke_streams:
        plans = plans[:a.smoke_streams]
    renderer = Renderer(device)
    current_obj = None
    files = {m: (a.out/(m+'.csv')).open('w') for m in methods}
    writers = {m: csv.DictWriter(f, fieldnames=FIELDS) for m, f in files.items()}
    for writer in writers.values():
        writer.writeheader()
    receipt = dict(completed=False, rank=a.rank, world=a.world,
                   protocol_sha256=sha(a.protocol), source_sha256=source_hash(),
                   smoke=bool(a.smoke_streams or a.smoke_frames),
                   streams=len(plans), completed_streams=0, processed_object_frames=0,
                   expected_predictions=len(expected_keys(plans)), predictions=0,
                   fp_calls=0, fp_modules_imported=False, gt_pose_reads=0, gt_mask_reads=0, hand_annotation_reads=0,
                   gt_resets=0, redetections_after_initialization=0,
                   failures={}, max_history_frames=0, max_ablated_history_frames=0,
                   device=str(device),fixture_only=bool(p.get('fixture_only',False)),
                   initializer_source=p.get('initializer_source','PoseCNN'),
                   max_valid_anchors=0,max_spatial_tokens_read=0,
                   primed_pose_references=0,ablated_pose_reference_updates=0)
    receipt['methods']=list(methods)
    if execution:
        receipt['execution_sha256'] = sha(a.execution)
    started = time.time()
    failures = Counter()
    def save():
        receipt.update(seconds=time.time()-started, failures=dict(failures))
        receipt['access_audit']=guard.snapshot()
        temp = a.out/'progress.tmp'
        temp.write_text(json.dumps(receipt, indent=2))
        temp.replace(a.out/'manifest.json')
    def lip_step(state, rgb, depth, timestamp, method):
        history = len(state.cache.metadata)
        if method == 'lip_no_feature_history':
            state = clear_feature_history(state)
            assert not state.cache.metadata and not state.cache.contexts
            visual=cache_summary(state.cache)
            assert visual['valid_anchors']==visual['dense_tokens']==0
            receipt['max_ablated_history_frames']=max(receipt['max_ablated_history_frames'],visual['recent_frames'])
            receipt['ablated_pose_reference_updates']+=int(visual['has_pose_reference'])
            history = 0
        else:
            receipt['max_history_frames'] = max(receipt['max_history_frames'], history)
        proposal, candidate = model.step(
            torch.from_numpy(rgb.transpose(2, 0, 1).copy()), torch.from_numpy(depth[None]),
            timestamp, state, renderer=renderer, precision=c['precision'],
            image_size=c['image_size'], crop_expansion=c['crop_expansion'])
        if method=='lip_temporal':
            receipt['max_valid_anchors']=max(receipt['max_valid_anchors'],cache_summary(candidate.cache)['valid_anchors'])
            receipt['max_spatial_tokens_read']=max(receipt['max_spatial_tokens_read'],int(proposal.get('spatial_tokens_read',0)))
        if proposal['status'] == 'ok':
            return model.commit(proposal, candidate), proposal['pose_original'].cpu().numpy(), 'ok', history
        failures[method+':'+proposal['status']] += 1
        return hold_failed_step(state, timestamp), proposal['pose_original'].cpu().numpy(), proposal['status'], history
    save()
    with torch.no_grad(), (a.out/'events.jsonl').open('w') as events:
        for plan in plans:
            scene_id, obj_id = plan['scene_id'], plan['obj_id']
            init = plan['initializer']
            if init is None:
                receipt['completed_streams'] += 1
                events.write(json.dumps(dict(scene_id=scene_id, obj_id=obj_id, status='no_initializer'))+'\n')
                save()
                continue
            if obj_id != current_obj:
                current_obj = obj_id
                path = raw_root/'models'/CLASSES[obj_id-1]/'textured_simple.obj'
                mesh, _ = mesh_metadata(path, Path(p['mesh_cache'])/str(a.rank), scale=1.)
            scene = raw_root/'bop/s0/test'/f'{scene_id:06d}'
            camera = json.loads((scene/'scene_camera.json').read_text())
            first, last = init['im_id'], plan['targets'][-1]['im_id']
            frames = list(range(first, last+1))
            assert all(str(f) in camera for f in frames), 'Missing intermediate frame calibration'
            if a.smoke_frames:
                frames = frames[:a.smoke_frames]
            target_frames = {t['im_id'] for t in plan['targets']}
            states, outputs = {}, {}
            for frame in frames:
                info = camera[str(frame)]
                K = np.asarray(info['cam_K'], dtype=np.float32).reshape(3, 3)
                rgb_path=scene/'rgb'/f'{frame:06d}.jpg';depth_path=scene/'depth'/f'{frame:06d}.png'
                guard.check(rgb_path,native=True);guard.check(depth_path,native=True)
                color = cv2.imread(str(rgb_path))
                dep = cv2.imread(str(depth_path), -1)
                if color is None or dep is None:
                    raise FileNotFoundError((scene_id, frame))
                rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                depth = dep.astype('f4')*float(info['depth_scale'])/1000.
                timestamp = frame/p['fps']
                status, history = {}, {}
                if frame == first:
                    common = init['pose'].astype(np.float32)
                    for method in methods:
                        outputs[method] = common.copy()
                        status[method] = 'initialized_external_fixture' if p.get('fixture_only') else 'initialized_posecnn_only'
                        state,warm_status=prime_initial_observation(model,common,mesh,K,f'{scene_id}:{obj_id}:{method}',
                            timestamp,p['fps'],depth.shape,obj_id,scene_id,
                            lambda state:lip_step(state,rgb,depth,timestamp,method))
                        states[method]=state
                        receipt['primed_pose_references']+=int(cache_summary(state.cache)['has_pose_reference'])
                        status[method] += ':'+warm_status
                        history[method] = 0
                else:
                    for method in methods:
                        assert torch.equal(states[method].K.cpu(), torch.from_numpy(K)), 'Changing camera K'
                        state, proposed, status[method], history[method] = lip_step(states[method], rgb, depth, timestamp, method)
                        states[method] = state
                        outputs[method] = proposed.copy()
                if frame in target_frames:
                    row = dict(scene_id=scene_id, im_id=frame, obj_id=obj_id, score=init['score'])
                    for method in methods:
                        writers[method].writerow(csv_row(row, outputs[method]))
                    receipt['predictions'] += 1
                receipt['processed_object_frames'] += 1
                events.write(json.dumps(dict(scene_id=scene_id, im_id=frame, obj_id=obj_id,
                    initialization=frame==first, target=frame in target_frames, status=status,
                    history_before=history, cache_after={m: len(s.cache.metadata) for m, s in states.items()},
                    cache_details_after={m:cache_summary(s.cache) for m,s in states.items()}))+'\n')
                if receipt['processed_object_frames'] % 100 == 0:
                    events.flush()
                    for f in files.values():
                        f.flush()
                    save()
            receipt['completed_streams'] += 1
            save()
            print(json.dumps({k: receipt[k] for k in ('rank', 'completed_streams', 'streams', 'processed_object_frames', 'predictions', 'seconds')}), flush=True)
    for f in files.values():
        f.close()
    if not receipt['smoke']:
        assert receipt['predictions'] == receipt['expected_predictions']
    assert 'estimater' not in sys.modules and not any(n.startswith('learning.training.predict_pose') for n in sys.modules)
    assert not any(value for key,value in guard.snapshot()['counts'].items() if key.startswith('denied_'))
    receipt.update(completed=True, csv_sha256={m: sha(a.out/(m+'.csv')) for m in methods})
    save()


if __name__ == '__main__':
    main()
