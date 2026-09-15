"""Bind a previously frozen checkpoint to the complete zero-FP s0 test protocol."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from lip.engine.config import check_data_gate
from lip.benchmark.bop_io import load_predictions
from streaming_bop_utils import plan_streams,expected_keys
from standalone_bop_state import standalone_methods


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('freeze','out','data-root','index-root','initializer','fp-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-initializer-sha',required=True)
    p.add_argument('--expected-targets-sha',required=True)
    p.add_argument('--methods',nargs='+',default=['lip_temporal'])
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    frozen=json.loads(a.freeze.read_text())
    assert frozen['completed'] and frozen['selected_before_test'] and frozen['test_access_before_selection'] is False
    checkpoint=Path(frozen['checkpoint']);assert sha(checkpoint)==frozen['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);audit=check_data_gate(a.index_root)
    assert state['split_hash']==audit['split_hash'] and state['mesh_hash']==audit['mesh_hash']
    methods=standalone_methods({'methods':a.methods})
    targets=a.data_root/'bop/s0/test_targets_bop19.json'
    assert sha(a.initializer)==a.expected_initializer_sha and sha(targets)==a.expected_targets_sha
    rows=json.loads(targets.read_text());plans=plan_streams(rows,load_predictions(a.initializer))
    population=dict(streams=len(plans),object_frames=sum(x['targets'][-1]['im_id']-x['initializer']['im_id']+1 for x in plans if x['initializer'] is not None),
        predictions_per_method=len(expected_keys(plans)),missing_targets=len(rows)-len(expected_keys(plans)),
        streams_without_initializer=sum(x['initializer'] is None for x in plans),target_rows=len(rows),
        target_images=len({(r['scene_id'],r['im_id']) for r in rows}))
    assert population==dict(streams=4916,object_frames=337862,predictions_per_method=87043,missing_targets=971,
        streams_without_initializer=38,target_rows=88014,target_images=23644)
    a.out.mkdir(parents=True,exist_ok=False);config=a.out/'config.yaml'
    config.write_text(yaml.safe_dump(dict(state['config'],precision='bf16'),sort_keys=False))
    protocol=dict(frozen=True,freeze_receipt=str(a.freeze.resolve()),freeze_sha256=sha(a.freeze),prepared=time.time(),
        split='s0_test',checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),
        architecture_id=state['architecture_id'],checkpoint_training_source_sha256=state['source_sha256'],
        source_sha256=source_hash(),config=str(config.resolve()),config_sha256=sha(config),
        initializer_csv=str(a.initializer.resolve()),initializer_sha256=a.expected_initializer_sha,
        initializer='First legal released PoseCNN pose directly; no GT pose, GT mask or FP.',
        targets=str(targets.resolve()),targets_sha256=a.expected_targets_sha,
        data_root=str(a.data_root.resolve()),index_root=str(a.index_root.resolve()),fp_root=str(a.fp_root.resolve()),
        mesh_cache=str((a.out/'mesh_cache').resolve()),history='causal_full_frames',fps=30.,methods=list(methods),
        population=population,fp_calls=0,no_gt_pose_or_mask_inputs=True,no_hand_annotations=True,
        protocol='Known-object causal RGB-D tracking on official s0 test targets with official BOP AR. This temporal-input protocol is not a single-image leaderboard submission.',
        frame_policy='Process every RGB-D frame from first legal PoseCNN detection through the last target; no future observations.',
        initialization_policy='Encode the initial RGB-D, discard that LIP proposal and retain the direct PoseCNN pose. Nonvisual reference state is corrected to the same external initializer.',
        history_ablation='If requested, clear only visual source/context/keyframe caches; retain accepted pose, motion, clock and nonvisual pose references.',
        failure_policy='Missing initializer targets stay absent in the official denominator. No GT reset or re-detection; invalid proposals hold the last pose and advance the clock. Software exceptions abort.',
        scope='All official target objects, with all/grasped aggregates reported separately. This official s0 target file contains 20 object IDs (1..18,20,21); the 21-class YCB catalog also contains ID19, which is absent from these targets.',
        official_single_image_comparable=False,inference_entrypoint='infer_lip_tracking_bop.py',
        tools_sha256={n:sha(root/'tools'/n) for n in ('infer_lip_tracking_bop.py','streaming_bop_utils.py','standalone_bop_state.py')},
        report_description=f"Frozen {frozen['name']}. Direct non-GT PoseCNN initialization, complete causal RGB-D history, no FoundationPose at initialization or tracking. Frozen before this test run; no test-driven checkpoint selection.")
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps(population))


if __name__=='__main__':main()
