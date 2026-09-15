"""Frozen causal RGB-D tracking; non-GT initialization; emit official target frames."""
import argparse
import csv
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import sys
import time
from collections import Counter
import cv2
import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lip.benchmark.bop_io import load_predictions, csv_row, FIELDS
from lip.engine.stream_checkpoint import sha, source_hash, load_init
from lip.engine.stream_config import load_stream_config, make_model
from lip.engine.config import check_data_gate
from lip.engine.stream_runtime import valid_pose
from lip.geometry.mesh import mesh_metadata
from lip.geometry.renderer import Renderer
from lip.data.index import CLASSES
from lip.integrations.foundationpose import FoundationPoseAdapter
from streaming_bop_utils import METHODS, plan_streams, expected_keys, clear_feature_history, hold_failed_step


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=8)
    parser.add_argument('--smoke-streams', type=int)
    parser.add_argument('--smoke-frames', type=int)
    parser.add_argument('--execution', type=Path)
    a = parser.parse_args()
    p = json.loads(a.protocol.read_text())
    assert p['frozen'] and p['history'] == 'causal_full_frames'
    assert source_hash() == p['source_sha256']
    for field in ('checkpoint', 'initializer_csv', 'targets', 'config'):
        digest_field = 'initializer_sha256' if field == 'initializer_csv' else field+'_sha256'
        assert sha(p[field]) == p[digest_field], field
    execution = json.loads(a.execution.read_text()) if a.execution else None
    if execution:
        assert execution['protocol_sha256'] == sha(a.protocol)
        assert execution['workers'] == a.world
        assert sha(execution['assignments']) == execution['assignments_sha256']
    for name, digest in (execution or p)['tools_sha256'].items():
        assert sha(Path(__file__).parent/name) == digest, name
    assert 0 <= a.rank < a.world
    raw_root = Path(p['data_root']).resolve()
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(raw_root):
            return
        if path.name.startswith(('labels_', 'scene_gt')) or any(x in ('mask', 'mask_visib') or x.startswith('mano') for x in path.parts):
            raise PermissionError('Inference annotation access: '+str(path))
        if args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
            raise PermissionError('Raw dataset is read-only')
    sys.addaudithook(guard)
    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    torch.manual_seed(42)
    np.random.seed(42)
    a.out.mkdir(parents=True, exist_ok=False)
    c = load_stream_config(p['config'])
    model = make_model(c).cuda().eval()
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
    sys.path.insert(0, p['fp_root'])
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    import nvdiffrast.torch as dr
    class RefineOnly(FoundationPose):
        def make_rotation_grid(self, *args, **kwargs):
            pass
    logging.getLogger().setLevel(logging.WARNING)
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    renderer = Renderer('cuda')
    estimator = None
    current_obj = None
    files = {m: (a.out/(m+'.csv')).open('w') for m in METHODS}
    writers = {m: csv.DictWriter(f, fieldnames=FIELDS) for m, f in files.items()}
    for writer in writers.values():
        writer.writeheader()
    receipt = dict(completed=False, rank=a.rank, world=a.world,
                   protocol_sha256=sha(a.protocol), source_sha256=source_hash(),
                   smoke=bool(a.smoke_streams or a.smoke_frames),
                   streams=len(plans), completed_streams=0, processed_object_frames=0,
                   expected_predictions=len(expected_keys(plans)), predictions=0,
                   gt_pose_reads=0, gt_mask_reads=0, hand_annotation_reads=0,
                   gt_resets=0, redetections_after_initialization=0,
                   failures={}, max_history_frames=0, max_ablated_history_frames=0)
    if execution:
        receipt['execution_sha256'] = sha(a.execution)
    started = time.time()
    failures = Counter()
    def save():
        receipt.update(seconds=time.time()-started, failures=dict(failures))
        temp = a.out/'progress.tmp'
        temp.write_text(json.dumps(receipt, indent=2))
        temp.replace(a.out/'manifest.json')
    def refine(prior, rgb, depth, K, tag):
        result = np.asarray(adapter.refine(prior, rgb, depth, K, iteration=p['fp_iterations']), dtype=np.float32)
        if not valid_pose(torch.from_numpy(result)):
            failures[tag+':invalid_pose'] += 1
            return np.asarray(prior, dtype=np.float32)
        return result
    def lip_step(state, rgb, depth, timestamp, method):
        history = len(state.cache.metadata)
        if method == 'lip_no_feature_history':
            state = clear_feature_history(state)
            assert not state.cache.metadata and not state.cache.contexts
            history = 0
        else:
            receipt['max_history_frames'] = max(receipt['max_history_frames'], history)
        proposal, candidate = model.step(
            torch.from_numpy(rgb.transpose(2, 0, 1).copy()), torch.from_numpy(depth[None]),
            timestamp, state, renderer=renderer, precision=c['precision'],
            image_size=c['image_size'], crop_expansion=c['crop_expansion'])
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
                raw = trimesh.load(path, process=False, force='mesh')
                if estimator is None:
                    estimator = RefineOnly(raw.vertices, raw.vertex_normals, mesh=raw, scorer=object(), refiner=refiner, glctx=glctx, debug=0, debug_dir=str(a.out/'fp_debug'))
                else:
                    estimator.reset_object(raw.vertices, raw.vertex_normals, mesh=raw)
                estimator.diameter = float(estimator.diameter)
                adapter = FoundationPoseAdapter(estimator)
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
                color = cv2.imread(str(scene/'rgb'/f'{frame:06d}.jpg'))
                dep = cv2.imread(str(scene/'depth'/f'{frame:06d}.png'), -1)
                if color is None or dep is None:
                    raise FileNotFoundError((scene_id, frame))
                rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                depth = dep.astype('f4')*float(info['depth_scale'])/1000.
                timestamp = frame/p['fps']
                status, history = {}, {}
                if frame == first:
                    common = refine(init['pose'], rgb, depth, K, 'common_initializer')
                    for method in METHODS:
                        outputs[method] = common.copy()
                        status[method] = 'initialized_common_posecnn_fp'
                        if method == 'fp_tracking':
                            continue
                        state = model.initialize(common, mesh, K, f'{scene_id}:{obj_id}:{method}',
                            timestamp-1/p['fps'], image_shape=depth.shape, object_id=obj_id, camera_id=scene_id)
                        # Encode the real initial observation; keep the SAME initial pose for all branches.
                        state, _, warm_status, _ = lip_step(state, rgb, depth, timestamp, method)
                        state = model.correct(state, common, timestamp=timestamp)
                        states[method] = replace(state, previous_pose=None, previous_timestamp=None)
                        status[method] += ':'+warm_status
                        history[method] = 0
                else:
                    outputs['fp_tracking'] = refine(outputs['fp_tracking'], rgb, depth, K, 'fp_tracking')
                    status['fp_tracking'] = 'ok'
                    for method in METHODS:
                        if method == 'fp_tracking':
                            continue
                        assert torch.equal(states[method].K.cpu(), torch.from_numpy(K)), 'Changing camera K'
                        state, proposed, status[method], history[method] = lip_step(states[method], rgb, depth, timestamp, method)
                        if method == 'lip_fp_temporal':
                            proposed = refine(proposed, rgb, depth, K, method)
                            old_cache = state.cache
                            state = model.correct(state, proposed, timestamp=timestamp)
                            assert state.cache is old_cache, 'FP correction discarded temporal features'
                        states[method] = state
                        outputs[method] = proposed.copy()
                if frame in target_frames:
                    row = dict(scene_id=scene_id, im_id=frame, obj_id=obj_id, score=init['score'])
                    for method in METHODS:
                        writers[method].writerow(csv_row(row, outputs[method]))
                    receipt['predictions'] += 1
                receipt['processed_object_frames'] += 1
                events.write(json.dumps(dict(scene_id=scene_id, im_id=frame, obj_id=obj_id,
                    initialization=frame==first, target=frame in target_frames, status=status,
                    history_before=history, cache_after={m: len(s.cache.metadata) for m, s in states.items()}))+'\n')
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
    receipt.update(completed=True, csv_sha256={m: sha(a.out/(m+'.csv')) for m in METHODS})
    save()


if __name__ == '__main__':
    main()
