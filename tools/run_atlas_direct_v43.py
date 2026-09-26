"""Matched100-update control and prior-free correspondence supervision."""
import json,signal,subprocess,sys,time
from pathlib import Path


def main():
    root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/atlas_direct_v43');root.mkdir(exist_ok=False)
    exe=Path(__file__).resolve().parents[1];child=None
    def status(stage,**kw):
        p=root/'status.tmp';p.write_text(json.dumps(dict(stage=stage,pid=child.pid if child else None,time=time.time(),**kw),indent=2));p.replace(root/'status.json')
    def stop(sig,_):
        if child and child.poll() is None:child.send_signal(signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for arm in ('assisted','direct'):
            with (root/f'{arm}.launch.log').open('w') as log:
                child=subprocess.Popen([sys.executable,'tools/run_geometry_transport_warmup.py','--config',f'configs/jepa/atlas_direct_v43_{arm}.yaml','--root',str(root/arm),'--steps','100','--clean-probe'],cwd=exe,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:status(arm);time.sleep(3)
                if child.returncode:raise RuntimeError(arm+' failed')
        status('complete',completed=True,updates_per_arm=100,source_step=100,default_model_changed=False)
    except Exception as e:status('failed',error=str(e));raise


if __name__=='__main__':main()
