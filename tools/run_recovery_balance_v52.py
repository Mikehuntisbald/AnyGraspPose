"""Two sequential, matched200-update recovery arms; no pose or adaptive budget."""
import json, os, signal, subprocess, sys, time
from pathlib import Path


def main():
    exe=Path(__file__).resolve().parents[1]
    root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/recovery_balance_v52')
    root.mkdir(exist_ok=False)
    child=None
    def status(stage,**kw):
        p=root/'status.tmp'
        p.write_text(json.dumps(dict(stage=stage,time=time.time(),pid=child.pid if child else None,**kw),indent=2))
        p.replace(root/'status.json')
    def stop(sig,_):
        if child and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=60)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for arm in ('control','balanced'):
            with (root/f'{arm}.launch.log').open('w') as log:
                child=subprocess.Popen([sys.executable,'tools/run_geometry_transport_warmup.py',
                    '--root',str(root/arm),'--config',f'configs/jepa/recovery_balance_v52_{arm}.yaml','--steps','200'],
                    cwd=exe,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                status(arm)
                if child.wait():raise RuntimeError(arm+' failed; remaining arms not launched')
        with (root/'confirmation.launch.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'tools/probe_recovery_confirmation_v52.py'],cwd=exe,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status('confirmation')
            if child.wait():raise RuntimeError('Confirmation failed')
        subprocess.run([sys.executable,'tools/report_recovery_balance_v52.py','--root',str(root)],cwd=exe,check=True)
        status('complete',completed=True,updates_per_arm=200,pose_training=False,pose_evaluation=False,default_model_changed=False)
    except Exception as e:
        if child and child.poll() is None:child.send_signal(signal.SIGTERM)
        status('failed',error=str(e));raise


if __name__=='__main__':main()
