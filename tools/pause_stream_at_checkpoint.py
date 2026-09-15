"""Archive a completed atomic checkpoint, then stop only its specified torchrun job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--architecture', default='stream_single', choices=['stream_single','stream_dual','stream_dual_cross','stream_dual_cross_residual'])
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=900)
    a = p.parse_args()
    import torch
    proc = Path('/proc') / str(a.pid)
    identity = (proc / 'stat').read_text().split()[21]
    command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    cwd = (proc / 'cwd').resolve()
    if 'torch.distributed.run' not in command or 'lip.train_stream' not in command:
        raise RuntimeError('Specified PID is not the streaming torchrun launcher')
    args = (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')
    if (cwd / args[args.index('--output') + 1]).resolve() != a.run.resolve():
        raise RuntimeError('Launcher output does not match the requested run')
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    if a.archive.exists():
        raise FileExistsError(a.archive)
    snapshot = a.archive.with_suffix('.pending.pt')
    if snapshot.exists():
        raise FileExistsError(snapshot)
    deadline = time.monotonic() + a.timeout
    checked_inode = None
    while time.monotonic() < deadline:
        if not proc.exists() or (proc / 'stat').read_text().split()[21] != identity:
            raise RuntimeError('Original launcher exited before the requested checkpoint')
        last = a.run / 'last.pt'
        inode = last.stat().st_ino if last.exists() else None
        if inode is not None and inode != checked_inode:
            # Link the exact published inode. Later atomic last.pt replacements
            # cannot change the checkpoint being inspected or archived.
            os.link(last, snapshot)
            saved = torch.load(snapshot, map_location='cpu', weights_only=False)
            step = saved['new_stage_step']
            print(json.dumps(dict(observed_checkpoint=step, target=a.step)), flush=True)
            if step >= a.step:
                if step != a.step:
                    raise RuntimeError('Requested checkpoint was already passed; no job stopped')
                assert saved['architecture_id'] == a.architecture
                assert saved['scheduler']['last_epoch'] == step
                assert len(saved['rng']) == 8 and saved['optimizer']['state']
                os.rename(snapshot, a.archive)
                digest = hashlib.sha256(a.archive.read_bytes()).hexdigest()
                receipt = dict(checkpoint=str(a.archive.resolve()), checkpoint_sha256=digest,
                               saved_step=step, sampler_position=saved['sampler_position'],
                               optimizer_and_rng_retained=True, launcher_pid=a.pid,
                               launcher_command=command, source_sha256=saved['source_sha256'],
                               signal='SIGTERM', stopped=False, time=time.time())
                a.receipt.write_text(json.dumps(receipt, indent=2))
                if (proc / 'stat').read_text().split()[21] != identity:
                    raise RuntimeError('Launcher identity changed; no signal sent')
                os.kill(a.pid, signal.SIGTERM)
                for _ in range(60):
                    if not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z':
                        receipt['stopped'] = True
                        break
                    time.sleep(1)
                a.receipt.write_text(json.dumps(receipt, indent=2))
                print(json.dumps(receipt), flush=True)
                if not receipt['stopped']:
                    raise RuntimeError('SIGTERM sent but launcher has not exited; inspect manually')
                return
            del saved
            snapshot.unlink()
            checked_inode = inode
        time.sleep(1)
    raise TimeoutError('No matching checkpoint; no process stopped')


if __name__ == '__main__':
    main()
