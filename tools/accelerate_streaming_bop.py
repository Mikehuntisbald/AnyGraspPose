"""Retain verified complete streams, then repartition unfinished streams over 4 workers/GPU."""
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha
from lip.benchmark.bop_io import load_predictions, FIELDS
from streaming_bop_utils import METHODS, plan_streams, expected_keys


def identity(plan):
    return plan['scene_id'], plan['obj_id']


def frame_count(plan):
    if plan['initializer'] is None:
        return 0
    return plan['targets'][-1]['im_id']-plan['initializer']['im_id']+1


def assign_balanced(plans, workers):
    assigned = {str(i): [] for i in range(workers)}
    loads = [0]*workers
    for plan in sorted(plans, key=lambda p: (-frame_count(p), identity(p))):
        worker = min(range(workers), key=lambda i: (loads[i], len(assigned[str(i)]), i))
        assigned[str(worker)].append(list(identity(plan)))
        loads[worker] += frame_count(plan)
    return assigned, loads


def main():
    root = Path(__file__).resolve().parents[1]
    run = root/'runs/streaming_s0_test'
    protocol = json.loads((run/'protocol.json').read_text())
    old_launch = json.loads((run/'launch.json').read_text())
    old_supervisor = json.loads((run/'supervisor.json').read_text())
    assert len(old_launch) == 8 and not (run/'execution_32.json').exists()
    assert json.loads((root/'runs/concurrency_probe/equivalence.json').read_text())['exported_poses_exactly_equal']
    plans = plan_streams(json.loads(Path(protocol['targets']).read_text()), load_predictions(protocol['initializer_csv']))
    paused, started = [], []
    committed = False
    def owned(pid, expected):
        assert expected.encode() in Path(f'/proc/{pid}/cmdline').read_bytes(), (pid, expected)
    try:
        owned(old_supervisor['pid'], 'finish_streaming_bop.py')
        os.kill(old_supervisor['pid'], signal.SIGSTOP)
        paused.append(old_supervisor['pid'])
        for item in old_launch:
            owned(item['pid'], 'infer_streaming_bop.py')
            os.kill(item['pid'], signal.SIGSTOP)
            paused.append(item['pid'])
        (run/'launch_original_8.json').write_text(json.dumps(old_launch, indent=2))
        (run/'supervisor_original_8.json').write_text(json.dumps(old_supervisor, indent=2))
        (run/'status.json').write_text(json.dumps(dict(phase='inference_repartitioning', started=time.time())))
        retained, retained_entries = set(), []
        old_frames = 0
        for item in old_launch:
            rank = item['rank']
            folder = run/f'rank{rank}'
            m = json.loads((folder/'manifest.json').read_text())
            assert not m['failures'], 'Explicit failure attribution needed before retaining a failing prefix'
            old_frames += m['processed_object_frames']
            ordered = sorted([p for p in plans if p['scene_id'] % 8 == rank], key=lambda p:(p['obj_id'],p['scene_id']))
            eligible = ordered[:m['completed_streams']]
            rows, keys = {}, {}
            for method in METHODS:
                data = (folder/(method+'.csv')).read_text()
                if not data.endswith('\n'):
                    data = data[:data.rfind('\n')+1]
                rows[method] = list(csv.DictReader(io.StringIO(data)))
                keys[method] = {(int(r['scene_id']),int(r['im_id']),int(r['obj_id'])) for r in rows[method]}
            complete = [p for p in eligible if all(expected_keys([p]) <= keys[method] for method in METHODS)]
            ids = {identity(p) for p in complete}
            assert not retained.intersection(ids)
            retained.update(ids)
            expected = expected_keys(complete)
            destination = run/f'retained{rank}'
            destination.mkdir(exist_ok=False)
            for method in METHODS:
                selected = [r for r in rows[method] if (int(r['scene_id']),int(r['obj_id'])) in ids]
                actual = [(int(r['scene_id']),int(r['im_id']),int(r['obj_id'])) for r in selected]
                assert len(actual) == len(set(actual)) and set(actual) == expected
                with (destination/(method+'.csv')).open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(selected)
            receipt = dict(m, completed=True, rank=f'retained{rank}', smoke=False,
                streams=len(complete), completed_streams=len(complete),
                processed_object_frames=sum(map(frame_count,complete)), predictions=len(expected),
                expected_predictions=len(expected), original_shard=str(folder),
                retained_complete_streams=[list(identity(p)) for p in complete],
                csv_sha256={method:sha(destination/(method+'.csv')) for method in METHODS})
            (destination/'manifest.json').write_text(json.dumps(receipt,indent=2))
            retained_entries.append(dict(rank=f'retained{rank}', folder=destination.name, retained=True))
        remaining = [p for p in plans if identity(p) not in retained]
        assignment, loads = assign_balanced(remaining, 32)
        flat = [tuple(pair) for pairs in assignment.values() for pair in pairs]
        assert len(flat) == len(set(flat)) and set(flat) == {identity(p) for p in remaining}
        retained_frames = sum(frame_count(p) for p in plans if identity(p) in retained)
        assert retained_frames + sum(loads) == protocol['population']['object_frames']
        assignment_file = run/'assignments_32.json'
        assignment_file.write_text(json.dumps(assignment,indent=2))
        execution = dict(protocol_sha256=sha(run/'protocol.json'),workers=32,workers_per_gpu=4,
            official_scoring_workers_per_method=16,
            assignments=str(assignment_file),assignments_sha256=sha(assignment_file),
            tools_sha256={n:sha(root/'tools'/n) for n in ('infer_streaming_bop.py','streaming_bop_utils.py')},
            retained_streams=len(retained), remaining_streams=len(remaining),
            retained_object_frames=retained_frames, remaining_object_frames=sum(loads),
            boundary_frames_to_redo=max(0,old_frames-retained_frames), worker_frame_loads=loads,
            control_tools_sha256={n:sha(root/'tools'/n) for n in ('accelerate_streaming_bop.py','finish_streaming_bop.py')},
            inference_math_changed=False, initialization_and_history_unchanged=True)
        (run/'execution_32.json').write_text(json.dumps(execution,indent=2))
        launch = list(retained_entries)
        for rank in range(32):
            folder = f'worker{rank}'
            cmd = [sys.executable,str(root/'tools/infer_streaming_bop.py'),'--protocol',str(run/'protocol.json'),
                '--execution',str(run/'execution_32.json'),'--out',str(run/folder),'--rank',str(rank),'--world','32']
            with (run/(folder+'.log')).open('w') as log:
                proc = subprocess.Popen(cmd,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank%8),
                    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(root/'src')),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started.append(proc.pid)
            launch.append(dict(rank=rank,folder=folder,pid=proc.pid,gpu=rank%8,command=cmd))
        temp = run/'launch.new.json'
        temp.write_text(json.dumps(launch,indent=2))
        temp.replace(run/'launch.json')
        cmd = old_supervisor['command']
        with (run/'supervisor_32.log').open('w') as log:
            proc = subprocess.Popen(cmd,cwd=root,env=dict(os.environ,PYTHONPATH=str(root/'src')),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started.append(proc.pid)
        (run/'supervisor.json').write_text(json.dumps(dict(pid=proc.pid,command=cmd),indent=2))
        committed = True
        for pid in paused:
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
        print(json.dumps(execution,indent=2))
    finally:
        if not committed:
            for pid in started:
                try: os.kill(pid, signal.SIGTERM)
                except ProcessLookupError: pass
            for pid in paused:
                os.kill(pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
