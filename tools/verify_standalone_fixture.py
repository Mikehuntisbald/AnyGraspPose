"""Verify completed train-only interface runs; this cannot approve a test GPU launch."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from lip.benchmark.bop_io import load_predictions


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--tests',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];fixture=json.loads((a.fixture/'fixture.json').read_text())
    assert fixture['completed'] and fixture['fixture_only'] and fixture['native_split']=='train'
    xml=ET.parse(a.tests).getroot();suites=list(xml.iter('testsuite'))
    assert suites and all(int(s.get('failures','0'))==int(s.get('errors','0'))==0 for s in suites)
    cases={}
    for name,entry in fixture['configurations'].items():
        folder=a.fixture/name;manifests={}
        for mode in ('both','single'):
            out=folder/mode;m=json.loads((out/'manifest.json').read_text());manifests[mode]=m
            assert m['completed'] and m['fixture_only'] and m['device']=='cpu' and not m['failures']
            assert m['source_sha256']==source_hash() and m['fp_calls']==m['gt_pose_reads']==m['gt_mask_reads']==m['hand_annotation_reads']==0
            assert not m['fp_modules_imported'] and m['processed_object_frames']==fixture['expected_processed_object_frames']
            assert m['predictions']==m['expected_predictions']==len(fixture['expected_target_keys'])
            assert m['max_history_frames']==8 and m['max_ablated_history_frames']==0
            counts=m['access_audit']['counts'];assert not any(v for k,v in counts.items() if k.startswith('denied_'))
            assert counts['validated_native_image_reads']==2*m['processed_object_frames']
            assert m['csv_sha256']['lip_temporal']==sha(out/'lip_temporal.csv')
            rows=load_predictions(out/'lip_temporal.csv')
            assert [[r['scene_id'],r['im_id'],r['obj_id']] for r in rows]==fixture['expected_target_keys']
            np.testing.assert_allclose(rows[0]['pose'],np.asarray(fixture['synthetic_initial_pose'],dtype='f4'),rtol=0,atol=1e-12)
            events=list(map(json.loads,(out/'events.jsonl').read_text().splitlines()))
            assert sum(r.get('status')=='no_initializer' for r in events)==1
            observed=[r for r in events if 'im_id' in r]
            assert [r['im_id'] for r in observed]==list(range(2,2+fixture['expected_processed_object_frames']))
            assert sum(r['initialization'] for r in observed)==1
        assert (folder/'both/lip_temporal.csv').read_bytes()==(folder/'single/lip_temporal.csv').read_bytes()
        cases[name]=dict(checkpoint_sha256=entry['checkpoint_sha256'],frames=fixture['expected_processed_object_frames'],
            single_and_two_method_csv_bitwise_equal=True,manifests={mode:sha(folder/mode/'manifest.json') for mode in manifests})
    receipt=dict(completed=True,fixture_only=True,source_sha256=source_hash(),cases=cases,
        tests=dict(xml=str(a.tests.resolve()),sha256=sha(a.tests),passed=sum(int(s.get('tests','0'))-int(s.get('skipped','0')) for s in suites)),
        tools_sha256={n:sha(root/'tools'/n) for n in ('infer_lip_tracking_bop.py','streaming_bop_utils.py','standalone_bop_state.py','verify_standalone_fixture.py')},
        scope='Real train RGB-D, synthetic non-GT initializer and missing-initialization canaries. No test accuracy or official GPU smoke claim. Full temporal CSV is unchanged when the unrequested feature-history ablation is omitted.')
    a.out.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
