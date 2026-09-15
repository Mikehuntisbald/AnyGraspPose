"""Finish event analysis on the existing augmented runs, without launching training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);p.add_argument('--controller-pid',required=True,type=int);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    experiment=a.experiment.resolve();out=a.out.resolve();root=Path(__file__).resolve().parents[1];out.mkdir(parents=True,exist_ok=False)
    def save(**value):
        temp=out/'status.tmp';temp.write_text(json.dumps(dict(pid=os.getpid(),controller_pid=a.controller_pid,updated=time.time(),**value),indent=2));temp.replace(out/'status.json')
    try:
        while True:
            status=json.loads((experiment/'status.json').read_text())
            if status['phase']=='completed':break
            if status['phase']=='failed':raise RuntimeError('Original training/evaluation failed: '+str(status.get('error')))
            proc=Path(f'/proc/{a.controller_pid}');cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'tools/run_spatial_memory.py' not in cmd or str(experiment) not in cmd:raise RuntimeError('Original controller identity is missing')
            if (proc/'stat').read_text().split(') ',1)[1].startswith('Z'):raise RuntimeError('Original controller ended without completion')
            save(phase='waiting_on_existing_controller',controller_phase=status['phase']);time.sleep(10)
        save(phase='analyzing');protocol=json.loads((experiment/'augmentation_protocol.json').read_text());e=json.loads((experiment/'experiment.json').read_text());old=Path(protocol['unaugmented_experiment'])
        folders=dict(parent=e['reference_evaluation'],M0A0=str(old/'control/s0_val'),M1A0=str(old/'spatial/s0_val'),M0A1=str(experiment/'control/s0_val'),M1A1=str(experiment/'spatial/s0_val'))
        flags=[v for name,path in folders.items() for v in ('--evaluation',name+'='+path)]
        commands=[('events',[sys.executable,str(root/'tools/diagnose_occlusion_events.py'),*flags,'--out',str(out/'events')]),
            ('figures',[sys.executable,str(root/'tools/plot_occlusion_event_cases.py'),'--events',str(out/'events/events.json'),*flags,'--index-root',e['index_root'],'--data-root',e['data_root'],'--out',str(out/'figures')])]
        for name,command in commands:
            with (out/(name+'.log')).open('w') as log:subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert json.loads((out/'events/events.json').read_text())['completed'];save(phase='completed',report=str(out/'events/report.md'),figures=str(out/'figures/figures.json'))
    except BaseException as error:save(phase='failed',error=repr(error));raise


if __name__=='__main__':main()
