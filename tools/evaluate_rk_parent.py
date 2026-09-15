"""Same-initializer parent reference; GPU sharing is explicit and scoped."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha
from lip.evaluate_stream import merge_shards


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);p.add_argument('--share-own-evaluation-gpus',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];r=a.experiment.resolve();e=json.loads((r/'experiment.json').read_text());request=json.loads((r/'evaluation_request.json').read_text())
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active:
        if not a.share_own_evaluation_gpus:raise RuntimeError('Explicit GPU sharing is required')
        allowed={c['pid'] for c in json.loads((r/'evaluation_status.json').read_text())['commands']}
        if set(map(int,active.splitlines()))-allowed:raise RuntimeError('Unrelated GPU job present')
    init=torch.load(e['arms']['R0K0']['init'],map_location='cpu',weights_only=False)
    parent=Path(init['parent']['path']);assert sha(parent)==e['parent_sha256']
    header=torch.load(parent,map_location='cpu',weights_only=False);cfg=header['config'];del header,init
    out=r/'parent_s0_val';out.mkdir(exist_ok=False);config=out/'config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    status=dict(phase='evaluating',checkpoint_sha256=sha(parent),initial_poses_sha256=request['initial_poses_sha256'],shared_gpus=bool(active),commands=[])
    jobs=[]
    try:
        for rank in range(8):
            command=[sys.executable,'-m','lip.evaluate_stream','--config',str(config),'--checkpoint',str(parent),'--out',str(out/f'rank{rank}'),
                '--data-root',e['data_root'],'--index-root',e['index_root'],'--split','val','--rank',str(rank),'--world-size','8','--initial-poses',request['initial_poses']]
            env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            log=(out/f'rank{rank}.log').open('w');proc=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log));status['commands'].append(dict(pid=proc.pid,gpu=rank,command=command))
        while any(proc.poll() is None for proc,log in jobs):
            if any(proc.poll() not in (None,0) for proc,log in jobs):raise RuntimeError('Parent shard failed')
            status['updated']=time.time();(out/'status.json').write_text(json.dumps(status,indent=2));time.sleep(5)
        for proc,log in jobs:log.close();assert proc.returncode==0
        merge_shards(out,8);status['phase']='completed'
    except BaseException as error:
        for proc,log in jobs:
            if proc.poll() is None:proc.terminate()
        for proc,log in jobs:proc.wait();log.close()
        status.update(phase='failed',error=repr(error));raise
    finally:(out/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
