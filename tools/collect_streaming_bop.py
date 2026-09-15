"""Refresh local progress and collect completed remote tracking results without predictions re-runs."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import time
import re

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--run-name',default='streaming_s0_test')
args=parser.parse_args()
assert re.fullmatch(r'[A-Za-z0-9_]+',args.run_name)
root = Path(__file__).resolve().parents[1]
out = root/'runs'/args.run_name
out.mkdir(parents=True,exist_ok=True)
remote = '/mnt/why/dexycb_lip/stream_bop_residual_1000_20260913/runs/'+args.run_name
ssh = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','10863','root@111.230.4.68']
while True:
    result = subprocess.run(ssh+['cat '+remote+'/status.json'], capture_output=True, text=True)
    if result.returncode:
        (out/'collector_error.log').write_text(result.stderr)
        time.sleep(30)
        continue
    status = json.loads(result.stdout)
    temp = out/'status.local.tmp'
    temp.write_text(json.dumps(status, indent=2))
    temp.replace(out/'status.json')
    if status['phase'] == 'failed':
        raise RuntimeError('Remote pipeline failed: '+status.get('error','inspect supervisor.log'))
    if status['phase'] == 'completed':
        # Selected final outputs only. Raw evaluator error/match shards stay remote.
        entries = ['protocol.json','comparison.json','occlusion_ar.json','report.md','status.json',
                   'inference_verified.json','dependency_receipt.json','supervisor.log','occlusion.log','csv']
        if (out/'execution_32.json').exists():
            entries += ['execution_32.json','assignments_32.json','launch.json','supervisor_32.log','throughput_32.json']
        methods=json.loads((out/'protocol.json').read_text())['methods']
        assert set(methods) <= {'lip_temporal','lip_no_feature_history','fp_tracking','lip_fp_temporal'}
        for method in methods:
            entries += [method+'/results.json', method+'/status.json', method+'/official_results.log',
                        method+'/bop-'+method.replace('_','-')+'_s0-test.csv']
        proc = subprocess.Popen(ssh+['tar -czf - -C '+remote+' '+' '.join(entries)], stdout=subprocess.PIPE)
        unpack = subprocess.run(['tar','-xzf','-','-C',str(out)], stdin=proc.stdout)
        proc.stdout.close()
        assert proc.wait() == 0 and unpack.returncode == 0
        assert hashlib.sha256((out/'occlusion_ar.json').read_bytes()).hexdigest() == status['occlusion_sha256']
        for method, digest in status['csv_sha256'].items():
            assert hashlib.sha256((out/'csv'/(method+'.csv')).read_bytes()).hexdigest() == digest
        (out/'local_collection_verified.json').write_text(json.dumps(dict(completed=True,
            occlusion_sha256=status['occlusion_sha256'], csv_sha256=status['csv_sha256']), indent=2))
        print('Completed artifacts collected and hashes verified', flush=True)
        break
    time.sleep(30)
