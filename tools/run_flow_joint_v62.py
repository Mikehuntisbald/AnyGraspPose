"""Bounded serial flow/JEPA prototype; 200 updates and paired causal probes."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


def main():
    exe = Path(__file__).resolve().parents[1]
    root = Path('/mnt/why/dexycb_lip/unified_jepa_20260921/flow_joint_v62')
    root.mkdir(exist_ok=False)
    config = 'configs/jepa/flow_joint_v62_control.yaml'
    out = root/'control/seed42'
    files = {str(p.relative_to(exe)):hashlib.sha256(p.read_bytes()).hexdigest()
             for d in ('src','tools','configs','tests') for p in (exe/d).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    (root/'source_receipt.json').write_text(json.dumps(files, indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as tar:
        for f in files:tar.add(exe/f,arcname=f)
    env = dict(os.environ, PYTHONPATH=str(exe/'src'), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    children = []
    def status(stage, **kw):
        p = root/'status.tmp'
        p.write_text(json.dumps(dict(stage=stage, pids=[p.pid for p in children], time=time.time(), **kw), indent=2))
        p.replace(root/'status.json')
    def stop(sig, _):
        for p in children:
            if p.poll() is None:os.killpg(p.pid, signal.SIGTERM)
        status('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM, stop);signal.signal(signal.SIGINT, stop)
    def run(stage, commands, sharded=False):
        nonlocal children
        assert all(hashlib.sha256((exe/f).read_bytes()).hexdigest()==h for f,h in files.items())
        children = []; logs = []
        try:
            for rank, command in enumerate(commands):
                log = (root/f'{stage}.{rank}.log').open('w');logs.append(log)
                children.append(subprocess.Popen([sys.executable]+command, cwd=exe,
                    env=dict(env, CUDA_VISIBLE_DEVICES=str(rank)) if sharded else env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None,0) for p in children):raise RuntimeError(stage+' failed')
                status(stage);time.sleep(3)
            if any(p.returncode for p in children):raise RuntimeError(stage+' failed')
        finally:
            for p in children:
                if p.poll() is None:os.killpg(p.pid, signal.SIGTERM)
            for log in logs:log.close()
    def probe(tag, checkpoint, cfg=config, ablation='on', angle=10, records=8):
        run(tag, [['tools/probe_geometry_transport.py','--config',cfg,'--checkpoint',str(checkpoint),
            '--out',str(root/'probe'/tag/f'rank{i}'),'--rank',str(i),'--world','8','--records',str(records),
            '--seed-start','62050000','--flow-ablation',ablation,'--rotation-degrees',str(angle)] for i in range(8)], True)
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPUs occupied')
        run('tests', [['-m','pytest','-q','tests/jepa/test_flow_reconstruction.py',
            'tests/jepa/test_cad_atlas_decoder.py','tests/jepa/test_canonical_surface_targets.py',
            'tests/jepa/test_supervision_quality.py','tests/jepa/test_execution_speed.py']])
        source='/mnt/why/dexycb_lip/unified_jepa_20260921/local_flow_v60_r1/training/extra.pt'
        source_config='/mnt/why/dexycb_lip/unified_jepa_20260921/local_flow_v60_r1/training/extra.yaml'
        for angle,records in ((0,4),(10,8),(60,4)):
            probe(f'baseline_{angle}',source,cfg=source_config,angle=angle,records=records)
        for arm in ('control','strong'):
            cfg=f'configs/jepa/flow_joint_v62_{arm}.yaml';out=root/arm/'seed42'
            for step in (2,200):
                command=['-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                    'tools/fp_worker.py','tools/train_geometry_transport.py','--config',cfg,'--stop-at',str(step)]
                if step==200:command+=['--resume',str(out/'last.pt')]
                run(f'{arm}_train{step}',[command])
            for angle,records in ((0,4),(10,8),(60,4)):
                probe(f'{arm}_{angle}',out/'last.pt',cfg=cfg,angle=angle,records=records)
        status('complete', completed=True, updates_per_arm=200, default_model_changed=False)
    except Exception as e:
        status('failed', error=str(e));raise


if __name__ == '__main__':main()
