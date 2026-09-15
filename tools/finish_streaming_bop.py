"""Persistently verify full causal inference, run official scoring and report."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha
from lip.benchmark.bop_io import load_predictions
from streaming_bop_utils import METHODS, plan_streams, expected_keys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--toolkit', type=Path, required=True)
    p.add_argument('--bop-python', required=True)
    p.add_argument('--resume-scoring', action='store_true')
    a = p.parse_args()
    out = a.run.resolve()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((out/'protocol.json').read_text())
    methods = tuple(protocol.get('methods', METHODS))
    launch = json.loads((out/'launch.json').read_text())
    receipt = dict(phase='inference', protocol_sha256=sha(out/'protocol.json'))
    def save():
        temp = out/'status.tmp'
        temp.write_text(json.dumps(receipt, indent=2))
        temp.replace(out/'status.json')
    try:
        if not a.resume_scoring:
            while True:
                manifests = []
                for item in launch:
                    folder = item.get('folder', f"rank{item['rank']}")
                    f = out/folder/'manifest.json'
                    m = json.loads(f.read_text()) if f.exists() else {}
                    manifests.append(m)
                    if not m.get('completed'):
                        cmd = Path(f"/proc/{item['pid']}/cmdline")
                        if not cmd.exists() or protocol.get('inference_entrypoint', 'infer_streaming_bop.py').encode() not in cmd.read_bytes():
                            raise RuntimeError(f"Rank {item['rank']} exited without completion")
                receipt.update(processed_object_frames=sum(m.get('processed_object_frames', 0) for m in manifests),
                               expected_object_frames=protocol['population']['object_frames'],
                               completed_streams=sum(m.get('completed_streams', 0) for m in manifests),
                               predictions_per_method=sum(m.get('predictions', 0) for m in manifests),
                               completed_ranks=sum(bool(m.get('completed')) for m in manifests))
                save()
                if all(m.get('completed') for m in manifests):
                    break
                time.sleep(20)
            assert all(m['protocol_sha256'] == receipt['protocol_sha256'] and not m['smoke'] for m in manifests)
            if protocol.get('fp_calls')==0:
                for m in manifests:
                    assert m['source_sha256']==protocol['source_sha256'] and m['fp_calls']==0 and not m['fp_modules_imported']
                    counts=m['access_audit']['counts']
                    assert not any(v for k,v in counts.items() if k.startswith('denied_'))
                    assert counts['validated_native_image_reads']==2*m['processed_object_frames']
                    assert m['gt_pose_reads']==m['gt_mask_reads']==m['hand_annotation_reads']==m['gt_resets']==0
            assert receipt['processed_object_frames'] == protocol['population']['object_frames']
            assert all(('lip_temporal' not in methods or m['max_history_frames'] == 8) and m['max_ablated_history_frames'] == 0 for m in manifests)
            plans = plan_streams(json.loads(Path(protocol['targets']).read_text()), load_predictions(protocol['initializer_csv']))
            expected = expected_keys(plans)
            merged = out/'csv'
            merged.mkdir(exist_ok=False)
            for method in methods:
                rows = []
                for item,m in zip(launch, manifests):
                    path = out/item.get('folder', f"rank{item['rank']}")/(method+'.csv')
                    assert sha(path) == m['csv_sha256'][method]
                    with path.open() as f:
                        rows.extend(csv.DictReader(f))
                keys = [(int(r['scene_id']), int(r['im_id']), int(r['obj_id'])) for r in rows]
                assert len(keys) == len(set(keys)) and set(keys) == expected
                rows.sort(key=lambda r: (int(r['scene_id']), int(r['im_id']), int(r['obj_id'])))
                with (merged/(method+'.csv')).open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=['scene_id','im_id','obj_id','score','R','t','time'])
                    writer.writeheader()
                    writer.writerows(rows)
            receipt.update(phase='official_evaluation', csv_sha256={m:sha(merged/(m+'.csv')) for m in methods},
                           population_verified=True, failures={str(m['rank']):m['failures'] for m in manifests})
            (out/'inference_verified.json').write_text(json.dumps(receipt, indent=2))
            save()
        else:
            prior_status = json.loads((out/'status.json').read_text())
            receipt = json.loads((out/'inference_verified.json').read_text())
            assert receipt['population_verified']
            merged = out/'csv'
            for method, digest in receipt['csv_sha256'].items():
                assert sha(merged/(method+'.csv')) == digest
            receipt.update(phase='official_evaluation', recovered_from=prior_status.get('error'),
                           scoring_wrapper_sha256=sha(root/'tools/evaluate_official_bop.py'))
            save()
        jobs = []
        for method in methods:
            destination = out/method
            if (destination/'results.json').exists():
                assert json.loads((destination/'status.json').read_text())['completed']
                receipt.setdefault('retained_completed_methods', []).append(method)
                save()
                continue
            if destination.exists():
                assert json.loads((destination/'status.json').read_text())['phase'] == 'failed'
                destination.rename(out/(method+'_failed_xvfb_'+str(int(time.time()))))
            cmd = [a.bop_python, str(root/'tools/evaluate_official_bop.py'), '--toolkit', str(a.toolkit),
                   '--data-root', protocol['data_root'], '--csv', str(merged/(method+'.csv')),
                   '--out', str(out/method), '--workers', '16']
            log = (out/(method+'_evaluation.log')).open('w')
            proc = subprocess.Popen(cmd, cwd=root, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((method, proc, log))
            receipt.setdefault('evaluation_jobs', []).append(dict(method=method, pid=proc.pid, command=cmd))
            save()
        for method, proc, log in jobs:
            code = proc.wait()
            log.close()
            if code:
                raise RuntimeError('Official scoring failed: '+method)
        results = {m:json.loads((out/m/'results.json').read_text()) for m in methods}
        (out/'comparison.json').write_text(json.dumps(dict(completed=True, protocol=protocol, results=results), indent=2))
        receipt['phase'] = 'occlusion_and_paired_intervals'
        save()
        with (out/'occlusion.log').open('w') as log:
            subprocess.run([a.bop_python, str(root/'tools/score_streaming_occlusion.py'), '--run', str(out), '--toolkit', str(a.toolkit)],
                           cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        occ = json.loads((out/'occlusion_ar.json').read_text())
        lines = ['# Full causal LIP evaluation on DexYCB s0 test', '',
                 protocol.get('report_description', 'Frozen residual 1,000. Non-GT PoseCNN→FP common initialization; each method then tracks its own state on every RGB-D frame. No GT resets or future observations. Official target frames and official BOP scoring; temporal input protocol is separate from single-image leaderboard conditions.'), '',
                 '| Method | All AR | Grasped AR | Visibility <0.5 AR | Visibility <0.3 AR |', '|---|---:|---:|---:|---:|']
        for method, r in results.items():
            o = occ['methods'][method]
            lines.append(f"| {method} | {r['all']['mean']:.4f} | {r['grasp_only']['mean']:.4f} | {o['all/v_lt_0.5']['AR']:.4f} | {o['all/v_lt_0.3']['AR']:.4f} |")
        lines += ['', 'Scores are percentages. BOP visibility buckets are posthoc diagnostics; they do not reproduce the Visibility Aware paper hand-model visibility or ADD AUC. Missing initialization targets remain in the official denominator.', '',
                  'All/grasped subsets and the empty <0.1 bucket are explicit in occlusion_ar.json.']
        if 'lip_no_feature_history' in methods:
            lines += ['lip_no_feature_history retains accepted pose/motion and nonvisual references, clearing only feature caches. Paired differences and 95% physical-sequence bootstrap intervals are in occlusion_ar.json.']
        (out/'report.md').write_text('\n'.join(lines)+'\n')
        receipt.update(phase='completed', completed=True, occlusion_sha256=sha(out/'occlusion_ar.json'))
        save()
    except BaseException as exc:
        receipt.update(phase='failed', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()
