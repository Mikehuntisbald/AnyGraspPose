"""Freeze protocol and audit causal population before any test predictions."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha, source_hash
from lip.benchmark.bop_io import load_predictions
from streaming_bop_utils import plan_streams, expected_keys, METHODS

root = Path(__file__).resolve().parents[1]
run = root/'runs/streaming_s0_test'
run.mkdir(exist_ok=False)
old = Path('/mnt/why/dexycb_lip/official_bop_residual_1000/runs/official_s0_test/protocol.json')
p = json.loads(old.read_text())
p.update(config=str(root/'configs/stream_lip_v2_residual_8800.yaml'),
         mesh_cache=str(root/'cache/mesh'), history='causal_full_frames', fps=30.,
         methods=list(METHODS), source_sha256=source_hash(),
         initializer='First available legal PoseCNN prediction, followed by two FP refinements shared by every branch',
         protocol='Known-object causal RGB-D tracking on s0 test. Official targets scored with official BOP evaluator; temporal inputs differ from single-image leaderboard conditions.',
         frame_policy='Process every RGB-D frame from first detection through the last target frame, including intermediate/non-target/occluded frames. No future pixels or poses.',
         initialization_policy='All methods emit identical common pose at initialization. LIP encodes the real initial observation and retains its features; initialization prediction is discarded. No motion is fabricated for the next frame.',
         failure_policy='Missing detections before initialization remain absent. No re-detection or GT reset after initialization. Illegal refinement retains its input; failed LIP update holds pose and advances clock. Software exceptions abort.',
         history_ablation='lip_no_feature_history clears source/context caches before each step, retaining previous pose, motion and clock. Each branch commits its own outputs.',
         fusion_history='LIP+FP commits LIP state then corrects pose with frozen FP, preserving source/context KV.',
         official_single_image_comparable=False,
         tools_sha256={n:sha(root/'tools'/n) for n in ('infer_streaming_bop.py','streaming_bop_utils.py')})
for k in ('inference_sha256','synthetic_dt','lip_updates_per_image','initializer_selected','initializer_missing'):
    p.pop(k, None)
p['config_sha256'] = sha(p['config'])
plans = plan_streams(json.loads(Path(p['targets']).read_text()), load_predictions(p['initializer_csv']))
frames = sum(x['targets'][-1]['im_id']-x['initializer']['im_id']+1 for x in plans if x['initializer'] is not None)
p['population'] = dict(streams=len(plans), object_frames=frames, predictions_per_method=len(expected_keys(plans)),
                       missing_targets=len(json.loads(Path(p['targets']).read_text()))-len(expected_keys(plans)),
                       streams_without_initializer=sum(x['initializer'] is None for x in plans),
                       target_rows=88014, target_images=23644)
(run/'protocol.json').write_text(json.dumps(p, indent=2)+'\n')
print(json.dumps(p['population'], indent=2))
