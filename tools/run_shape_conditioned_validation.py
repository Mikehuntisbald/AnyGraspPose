"""Serial eight-GPU controlled validation of three completed readout arms."""
import subprocess,os,sys,json,time,signal
from pathlib import Path

def main():
    root=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/shape_conditioned_v24');exe=Path(__file__).resolve().parents[1]
    parent=root.parent/'serial_completion_v21/runs/seed42/last.pt';children=[]
    def status(stage,**kw):
        temp=root/'val_status.tmp';temp.write_text(json.dumps(dict(stage=stage,pids=[p.pid for p in children],time=time.time(),**kw),indent=2));temp.replace(root/'val_status.json')
    def stop(sig,_):
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied')
        for arm in ('legacy','unit_gate','shape_conditioned'):
            rec=json.loads((root/arm/'receipt.json').read_text());assert rec['completed']
            children=[];logs=[]
            try:
                for rank in range(8):
                    receipt=root/arm/'validation'/f'rank{rank}'/'receipt.json'
                    if receipt.exists():
                        done=json.loads(receipt.read_text())
                        assert done['completed'] and done['readout_override']['sha256']==rec['checkpoint_sha256']
                        continue
                    log=(root/(arm+f'.val{rank}.log')).open('w');logs.append(log)
                    args=[sys.executable,'tools/probe_serial_pose.py','--config','configs/jepa/geometry_fidelity_v22_control.yaml','--checkpoint',str(parent),'--out',str(root/arm/'validation'/f'rank{rank}'),'--rank',str(rank),'--world','8','--geometry-components','--readout-checkpoint',str(root/arm/'readout.pt')]
                    children.append(subprocess.Popen(args,cwd=exe,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),PYTHONPATH=str(exe/'src'),OMP_NUM_THREADS='2'),start_new_session=True))
                while any(p.poll() is None for p in children):
                    if any(p.poll() not in (None,0) for p in children):raise RuntimeError(arm+' validation failed')
                    status(arm);time.sleep(5)
                if any(p.returncode for p in children):raise RuntimeError(arm+' validation failed')
                subprocess.run([sys.executable,'tools/report_serial_geometry_probe.py','--root',str(root/arm/'validation')],cwd=exe,stdout=(root/(arm+'.summary.log')).open('w'),check=True)
            finally:
                for p in children:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
                for log in logs:log.close()
        status('complete',completed=True)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
