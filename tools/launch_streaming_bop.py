"""Launch the already frozen, tested four-branch evaluation across eight GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
run = root/'runs/streaming_s0_test'
assert not (run/'launch.json').exists()
tests = ET.parse(root/'runs/protocol_tests.xml').getroot()
assert all(int(s.get('failures', 0))+int(s.get('errors', 0)) == 0 for s in tests.iter('testsuite'))
smoke = json.loads((root/'runs/smoke_verified/manifest.json').read_text())
assert smoke['completed'] and not smoke['failures'] and smoke['max_history_frames'] == 8
events = [json.loads(s) for s in (root/'runs/smoke_verified/events.jsonl').read_text().splitlines()]
assert len(events) == 32
assert all(e['history_before']['lip_no_feature_history'] == 0 for e in events)
assert max(e['history_before']['lip_fp_temporal'] for e in events) == 8
assert all(e['cache_after']['lip_fp_temporal'] == e['cache_after']['lip_temporal'] for e in events)
for name in ('protocol_tests.xml', 'protocol_tests.log', 'smoke_verified.log'):
    (run/name).write_bytes((root/'runs'/name).read_bytes())
(run/'smoke_manifest.json').write_text(json.dumps(smoke, indent=2))
launch = []
for rank in range(8):
    cmd = [sys.executable, str(root/'tools/infer_streaming_bop.py'), '--protocol', str(run/'protocol.json'),
           '--out', str(run/f'rank{rank}'), '--rank', str(rank), '--world', '8']
    with (run/f'rank{rank}.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank),
            OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONPATH=str(root/'src')),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch.append(dict(rank=rank, pid=proc.pid, command=cmd))
(run/'launch.json').write_text(json.dumps(launch, indent=2))
cmd = [sys.executable, str(root/'tools/finish_streaming_bop.py'), '--run', str(run),
       '--toolkit', '/mnt/why/dexycb_lip/third_party/dex-ycb-toolkit-official',
       '--bop-python', '/mnt/why/dexycb_lip/.venv-bop/bin/python']
with (run/'supervisor.log').open('w') as log:
    proc = subprocess.Popen(cmd, cwd=root, env=dict(os.environ, PYTHONPATH=str(root/'src')),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(run/'supervisor.json').write_text(json.dumps(dict(pid=proc.pid, command=cmd), indent=2))
print(json.dumps(dict(inference=launch, supervisor_pid=proc.pid), indent=2))
