"""Authorized remote workflow: verified upload -> complete s0 preflight -> training.

Run under tmux. All work stays in this project; raw archives are read-only.
Failures stop the chain, retain partial artifacts and write status.json.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def validate_upload(expected, receipt, archive_root):
    archive_root = Path(archive_root).resolve()
    if receipt.get('status') != 'complete':
        raise ValueError('Upload receipt does not report completion')
    targets = {r['name']: r for r in expected['files']}
    if len(targets) != len(expected['files']):
        raise ValueError('Duplicate expected archive')
    actual = receipt.get('files', [])
    names = [r['name'] for r in actual]
    if len(names) != len(set(names)) or set(names) != set(targets):
        raise ValueError('Upload receipt is missing or duplicating archives')
    total = 0
    for row in actual:
        name = row['name']
        if Path(name).name != name:
            raise ValueError('Invalid archive name')
        src = archive_root / name
        if Path(row['remote_path']).resolve() != src:
            raise ValueError('Unexpected archive destination')
        sha = row.get('sha256', '')
        if not re.fullmatch('[0-9a-f]{64}', sha) or row.get('remote_sha256') != sha:
            raise ValueError('Missing or inconsistent SHA256')
        if row.get('status') != 'verified' or row.get('gzip_crc_verified') is not True:
            raise ValueError('Archive is not verified')
        if row['bytes'] != targets[name]['bytes'] or not src.is_file() or src.stat().st_size != row['bytes']:
            raise ValueError('Missing archive or size mismatch: ' + name)
        total += row['bytes']
    if total != expected['total_bytes'] or total != receipt['total_bytes']:
        raise ValueError('Total byte count mismatch')
    return dict(files=actual, scope='full official DexYCB: all 10 subjects plus bop/calibration/models',
                upload_finished_utc=receipt.get('finished_utc'))


def source_digest(root):
    h = hashlib.sha256()
    for directory in ['src', 'tools', 'scripts', 'tests']:
        for p in sorted((root / directory).rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.sh') and '__pycache__' not in p.parts:
                h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    p = root / 'configs/dexycb_lip_v1.yaml'
    h.update(p.read_bytes())
    return h.hexdigest()


class Workflow:
    def __init__(self, root, job, data_root, archive_root):
        self.root = Path(root).resolve()
        self.job = Path(job).resolve()
        self.data = Path(data_root).resolve()
        self.archives = Path(archive_root).resolve()
        if not self.job.is_relative_to(self.root) or not self.data.is_relative_to(self.root):
            raise ValueError('Job outputs and extraction must remain under the project')
        self.job.mkdir(parents=True, exist_ok=True)
        self.lock = (self.job / 'lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.child = None
        self.context = {}
        self.env = dict(os.environ, DEX_YCB_DIR=str(self.data), LIP_WORK_DIR=str(self.root),
                        PYTHONPATH=str(self.root / 'src'), PYTHONUNBUFFERED='1')

    def status(self, phase, **fields):
        value = dict(pid=os.getpid(), phase=phase, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     data_root=str(self.data), job_root=str(self.job), **self.context, **fields)
        atomic_json(self.job / 'status.json', value)
        return value

    def event(self, kind, **fields):
        print(json.dumps(dict(event=kind, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)), flush=True)

    def wait_upload(self):
        expected = json.loads((self.job / 'expected_upload.json').read_text())
        path = self.archives / 'upload_receipt.json'
        last_log = 0
        while not path.exists():
            complete = 0; present = 0
            for row in expected['files']:
                final = self.archives / row['name']
                partial = self.archives / '.upload' / (row['name'] + '.part')
                if final.is_file() and final.stat().st_size == row['bytes']:
                    complete += 1; present += row['bytes']
                elif partial.exists():
                    present += min(partial.stat().st_size, row['bytes'])
            value = self.status('waiting_upload', archives_present=complete, archives_expected=len(expected['files']),
                                bytes_present=present, total_bytes=expected['total_bytes'], training_started=False)
            if time.monotonic() - last_log >= 60:
                self.event('waiting_upload', archives_present=complete, bytes_present=present)
                last_log = time.monotonic()
            time.sleep(30)
        receipt = json.loads(path.read_text())
        manifest = validate_upload(expected, receipt, self.archives)
        atomic_json(self.job / 'verified_upload.json', manifest)
        required=dict(manifest,files=[r for r in manifest['files'] if r['name']!='bop.tar.gz'],
                      scope='All ten subjects, calibration and models; BOP alternate-format extraction is independent')
        atomic_json(self.job / 'training_archives.json', required)
        shutil.copy2(path, self.job / 'upload_receipt.json')
        self.context['upload_receipt_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.event('upload_verified', files=len(manifest['files']))

    def snapshot(self):
        digest = source_digest(self.root)
        mark = self.job / 'snapshot.json'
        if mark.exists():
            if json.loads(mark.read_text())['source_sha256'] != digest:
                raise RuntimeError('Source changed since this preflight snapshot; retained outputs require review before rerunning')
        else:
            dest = self.job / 'source_snapshot'
            for name in ['src', 'tools', 'scripts', 'tests', 'configs']:
                shutil.copytree(self.root / name, dest / name, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__'))
            atomic_json(mark, dict(source_sha256=digest, authorization='2026-09-10 user: 完成后开始训练',
                                   max_optimizer_steps=40000, world_size=8))
        self.context['source_sha256'] = digest

    def wait_gpus(self):
        while True:
            r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            rows = [line.split(', ') for line in r.stdout.strip().splitlines()]
            if len(rows) != 8 or any('H20' not in row[1] for row in rows):
                raise RuntimeError('Expected one machine with 8 H20 GPUs')
            apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if not apps.stdout.strip() and all(float(row[3]) < .1 * float(row[2]) for row in rows):
                atomic_json(self.job / 'gpu_before_preflight.json', dict(rows=rows)); return
            self.status('waiting_free_gpus', gpu_rows=rows, training_started=False)
            time.sleep(30)

    def run_stage(self, name, command, mirror=None, training=False):
        marker = self.job / (name + '.json')
        command = [str(x) for x in command]
        identity = dict(command=command, source_sha256=self.context['source_sha256'],
                        upload_receipt_sha256=self.context['upload_receipt_sha256'])
        if marker.exists():
            old = json.loads(marker.read_text())
            if old.get('status') == 'complete':
                if any(old.get(k) != v for k, v in identity.items()):
                    raise RuntimeError('Stale stage receipt: ' + name)
                self.event('stage_already_complete', stage=name); return
        logfile = self.job / (name + '.log')
        self.event('stage_start', stage=name, command=command)
        start = time.monotonic()
        with logfile.open('a') as log:
            self.child = subprocess.Popen(command, cwd=self.root, env=self.env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            while self.child.poll() is None:
                fields = dict(stage=name, child_pid=self.child.pid, log=str(logfile), training_started=training or name=='verify_training_completion')
                if training:
                    ranklog = self.root / 'runs/lip_v1_s0/rank0.jsonl'
                    if ranklog.exists():
                        with ranklog.open('rb') as f:
                            f.seek(max(0, ranklog.stat().st_size - 16384)); lines=f.read().splitlines()
                        for line in reversed(lines):
                            try:
                                record=json.loads(line);fields['optimizer_step']=record['step'];break
                            except (ValueError, KeyError): pass
                self.status('training' if training else 'preflight', **fields)
                time.sleep(5)
            code = self.child.returncode
        self.child = None
        if mirror is not None: shutil.copy2(logfile, self.root / mirror)
        result = dict(**identity, status='complete' if code == 0 else 'failed', returncode=code,
                      elapsed_seconds=time.monotonic()-start, log=str(logfile))
        atomic_json(marker, result)
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}; see {logfile}')
        self.event('stage_complete', stage=name, elapsed_seconds=result['elapsed_seconds'])

    def execute(self):
        self.wait_upload()
        self.snapshot()
        python = sys.executable
        self.run_stage('extract', [python, 'tools/prepare_available.py', '--manifest', self.job/'training_archives.json', '--out', self.data])
        self.run_stage('audit', [python, 'tools/audit_data.py', '--data-root', self.data, '--out', 'runs/audit'])
        self.run_stage('index', [python, 'tools/build_index.py', '--data-root', self.data, '--setup', 's0', '--out', 'cache/dexycb_s0'])
        self.wait_gpus()
        self.run_stage('geometry', [python, 'tools/check_geometry.py', '--data-root', self.data, '--index', 'cache/dexycb_s0', '--num-samples', '32', '--out', 'runs/geometry'])
        launch=[python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=8']
        self.run_stage('sampling', launch+['tools/build_sampling.py', '--data-root', self.data, '--index', 'cache/dexycb_s0', '--length', '8'])
        self.run_stage('merge_sampling', [python, 'tools/merge_sampling.py', '--index', 'cache/dexycb_s0', '--world', '8'])
        self.run_stage('pytest', [python, '-m', 'pytest', '-q'], mirror='runs/pytest_preflight.log')
        self.run_stage('overfit32', [python, 'tools/overfit_small.py', '--config', 'configs/dexycb_lip_v1.yaml', '--num-clips', '32', '--steps', '500'])
        self.run_stage('probe', [python, 'tools/probe_batch.py', '--config', 'configs/dexycb_lip_v1.yaml', '--out', 'configs/resolved_8gpu.yaml'])
        self.run_stage('ddp_smoke_resume', ['bash', 'scripts/smoke_8gpu.sh'])
        self.run_stage('approve_full_preflight', [python, 'tools/verify_preflight.py', '--config', 'configs/resolved_8gpu.yaml', '--approve'])
        self.wait_gpus()
        command=['bash', 'scripts/train_8gpu.sh']
        checkpoint=self.root/'runs/lip_v1_s0/last.pt'
        if checkpoint.exists(): command+=['--resume', str(checkpoint)]
        self.run_stage('train_40000', command, training=True)
        self.run_stage('verify_training_completion', [python,'tools/verify_training_completion.py','--expected-steps','40000'])
        self.status('complete', training_started=True, training_process_exited=True,
                    checkpoint=str(self.root/'runs/lip_v1_s0/last.pt'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--job',default='runs/full_train_after_upload_20260910')
    p.add_argument('--data-root',default='cache/raw_full_20260910')
    p.add_argument('--archive-root',default='/mnt/why/DexYCB')
    a=p.parse_args();root=Path(a.root).resolve()
    flow=Workflow(root,root/a.job,root/a.data_root,a.archive_root)
    def stop(signum,frame):
        if flow.child and flow.child.poll() is None: os.killpg(flow.child.pid,signal.SIGTERM)
        flow.status('interrupted',signal=signum)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try: flow.execute()
    except Exception as e:
        flow.status('failed',error=str(e));flow.event('failed',error=str(e));raise


if __name__=='__main__': main()
