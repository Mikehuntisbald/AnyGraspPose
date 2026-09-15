"""Collect and merge frozen train-only initialization predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('source','native-build','deps','checkpoint','data-root','index-root','requests','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-checkpoint-sha',required=True);p.add_argument('--gpus',default='4,5,6,7');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=False);gpus=a.gpus.split(',');jobs=[]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec=json.loads(a.requests.read_text());assert spec['split']=='train';keys={v['key'] for v in spec['requests']}
    files=[root/'tools'/n for n in ('infer_posecnn_train.py','posecnn_backend.py','val_non_gt_common.py','standalone_bop_state.py')]
    status=dict(phase='inference',started=time.time(),commands=[],requests_sha256=sha(a.requests),tools_sha256={str(f):sha(f) for f in files})
    def save():
        status['updated']=time.time();tmp=a.out/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(a.out/'status.json')
    try:
        for rank,gpu in enumerate(gpus):
            cmd=[sys.executable,'tools/infer_posecnn_train.py']
            for n in ('source','native-build','deps','checkpoint','data-root','index-root','requests'):cmd+=['--'+n,str(getattr(a,n.replace('-','_')))]
            cmd+=['--expected-checkpoint-sha',a.expected_checkpoint_sha,'--rank',str(rank),'--world',str(len(gpus)),'--out',str(a.out/f'rank{rank}')]
            log=(a.out/f'rank{rank}.log').open('x');proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log));status['commands'].append(dict(rank=rank,pid=proc.pid,command=cmd));save()
        while any(p.poll() is None for p,f in jobs):
            if any(p.poll() not in (None,0) for p,f in jobs):raise RuntimeError('Train initializer worker failed')
            save();time.sleep(5)
        for p,f in jobs:f.close();assert p.returncode==0
        jobs=[];status['phase']='merging';save();manifests=[];values={}
        for rank in range(len(gpus)):
            folder=a.out/f'rank{rank}';m=json.loads((folder/'initializers.json').read_text());assert m['completed'] and not m['subset'] and m['split']=='train'
            assert m['uses_gt_pose'] is False and m['fp_calls']==0 and m['requests_sha256']==sha(a.requests)
            assert m['checkpoint_sha256']==a.expected_checkpoint_sha and sha(folder/'attempts.jsonl')==m['attempts_sha256']
            assert not any(v for k,v in m['access_audit']['counts'].items() if k.startswith('denied_'))
            assert m['access_audit']['counts']['validated_native_image_reads']==m['processed']==len(m['initializers'])
            assert not set(values)&set(m['initializers']);values.update(m['initializers']);manifests.append(m)
        assert set(values)==keys
        for f,digest in status['tools_sha256'].items():assert sha(f)==digest
        assert sha(a.requests)==status['requests_sha256']
        for key in ('inference_source_sha256','entrypoint_sha256','guard_sha256','split_hash','mesh_hash'):
            assert all(m[key]==manifests[0][key] for m in manifests),key
        result=dict(manifests[0],completed=True,rank=None,world=len(gpus),expected_requests=len(keys),processed=len(keys),
            initializers=values,initialized=sum(v is not None for v in values.values()),
            access_audit=dict(manifests[0]['access_audit'],counts={k:sum(m['access_audit']['counts'].get(k,0) for m in manifests) for k in set().union(*(m['access_audit']['counts'] for m in manifests))}),
            shards=[dict(rank=i,initializers_sha256=sha(a.out/f'rank{i}/initializers.json'),attempts_sha256=m['attempts_sha256']) for i,m in enumerate(manifests)])
        result.pop('attempts_sha256');(a.out/'initializers.json').write_text(json.dumps(result,indent=2))
        status.update(phase='completed',completed=time.time(),initializers_sha256=sha(a.out/'initializers.json'),initialized=result['initialized'],requests=len(keys));save()
    except BaseException as error:
        for p,f in jobs:
            if p.poll() is None:p.terminate()
        for p,f in jobs:p.wait();f.close()
        status.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
