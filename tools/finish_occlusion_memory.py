"""Wait on the existing controller, then produce the declared four-arm report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);p.add_argument('--controller-pid',required=True,type=int);a=p.parse_args()
    r=a.experiment.resolve();root=Path(__file__).resolve().parents[1];status=r/'factorial_finish_status.json'
    (r/'factorial_finish.lock').open('x').write(str(os.getpid()))
    def save(**value):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(updated=time.time(),pid=os.getpid(),controller_pid=a.controller_pid,**value),indent=2));temp.replace(status)
    try:
        while True:
            current=json.loads((r/'status.json').read_text())
            if current['phase']=='completed':break
            if current['phase']=='failed':raise RuntimeError('Existing training/evaluation controller failed: '+str(current.get('error')))
            proc=Path(f'/proc/{a.controller_pid}')
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'tools/run_spatial_memory.py' not in cmd or str(r) not in cmd:raise RuntimeError('Controller PID identity changed')
            if (proc/'stat').read_text().split(') ',1)[1].startswith('Z'):raise RuntimeError('Controller exited without a completed receipt')
            save(phase='waiting_for_existing_controller',controller_phase=current['phase']);time.sleep(10)
        save(phase='comparing')
        subprocess.run([sys.executable,str(root/'tools/compare_occlusion_memory.py'),'--experiment',str(r)],cwd=root,check=True)
        receipt=json.loads((r/'occlusion_comparison.json').read_text());assert receipt['completed']
        save(phase='completed',report=str(r/'occlusion_report.md'))
    except BaseException as error:
        save(phase='failed',error=repr(error));raise


if __name__=='__main__':main()
