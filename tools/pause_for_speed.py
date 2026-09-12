"""Preserve a requested atomic checkpoint before stopping the identified launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import torch

p = argparse.ArgumentParser()
p.add_argument('--launcher', type=int, required=True)
p.add_argument('--step', type=int, required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
source = Path(a.checkpoint); output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
cmd = Path(f'/proc/{a.launcher}/cmdline').read_bytes().replace(b'\0', b' ').decode()
assert 'torch.distributed.run' in cmd and 'lip.train' in cmd
last_mtime = None
while True:
    assert Path(f'/proc/{a.launcher}').exists(), 'Training exited before requested checkpoint'
    mtime = source.stat().st_mtime_ns
    if mtime != last_mtime:
        last_mtime = mtime
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        step = checkpoint['global_step']
        if step >= a.step:
            assert step == a.step, f'Missed requested checkpoint: {step}'
            saved = output / f'resume_{step}.pt'
            shutil.copy2(source, saved)
            assert torch.load(saved, map_location='cpu', weights_only=False)['global_step'] == step
            # Elastic launcher propagates TERM to its workers. Capture their
            # process groups first so prefetched loader children cannot leak.
            groups = []
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit(): continue
                try:
                    stat = (proc / 'stat').read_text().split(') ', 1)[1].split()
                    if int(stat[1]) == a.launcher: groups.append(int(proc.name))
                except (FileNotFoundError, ProcessLookupError): pass
            os.kill(a.launcher, signal.SIGTERM)
            for _ in range(30):
                if not Path(f'/proc/{a.launcher}').exists(): break
                time.sleep(1)
            for pgid in groups:
                try: os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError: pass
            receipt = dict(paused=True, step=step, checkpoint=str(saved),
                           checkpoint_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
                           previous_launcher=a.launcher, previous_command=cmd,
                           utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            (output/'paused.json').write_text(json.dumps(receipt, indent=2))
            break
        del checkpoint
    time.sleep(.5)
