"""Retain one exact optimizer step without changing or pausing the trainer."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import torch

p=argparse.ArgumentParser(__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--step',type=int,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
lock=(a.out/f'retain_{a.step}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(**kw):
    dst=a.out/f'retain_{a.step}_status.json';tmp=dst.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(requested_step=a.step,utc=time.time(),**kw),indent=2));tmp.replace(dst)
last_inode=None;deadline=time.monotonic()+14400
probe=a.out/f'retain_{a.step}_probe.pt'
try:
    while time.monotonic()<deadline:
        st=a.source.stat()
        if st.st_ino==last_inode:time.sleep(2);continue
        assert not probe.exists()
        os.link(a.source,probe);last_inode=probe.stat().st_ino
        ck=torch.load(probe,map_location='cpu',weights_only=False)
        step=ck['global_step'];assert step==ck['scheduler']['last_epoch']==ck['sampler_position'];del ck
        if step<a.step:
            probe.unlink();status(phase='waiting',last_seen_step=step);time.sleep(2);continue
        if step>a.step:
            probe.unlink();raise RuntimeError(f'Requested checkpoint already passed: observed {step}')
        dst=a.out/f'checkpoint_{step}.pt'
        assert not dst.exists(),str(dst)
        probe.rename(dst)
        record=dict(phase='retained',step=step,checkpoint=str(dst.resolve()),sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),source=str(a.source.resolve()),evaluation_started=False)
        (a.out/f'checkpoint_{step}.json').write_text(json.dumps(record,indent=2));status(**record)
        print(json.dumps(record),flush=True);break
    else:raise TimeoutError('Four-hour retention deadline reached')
except Exception as exc:
    status(phase='failed',error=repr(exc));raise
