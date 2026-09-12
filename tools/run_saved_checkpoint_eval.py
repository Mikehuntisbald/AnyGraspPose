"""Run the retained evaluator on a saved checkpoint and verify the reference population."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from lip.evaluation.metrics import summarize

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--reference',required=True)
    parser.add_argument('--step',required=True,type=int)
    parser.add_argument('--methods',nargs='+',choices=('lip','lip_fp'),default=['lip','lip_fp'])
    a=parser.parse_args()
    assert len(a.methods)==len(set(a.methods)), 'Duplicate evaluation method'
    root=Path('/mnt/why/dexycb_lip');os.chdir(root)
    job=Path(a.job).resolve();reference=Path(a.reference).resolve();checkpoint=Path(a.checkpoint).resolve()
    checkpoint_hash=sha(checkpoint);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['global_step']==ck['scheduler']['last_epoch']==ck['sampler_position']==a.step
    del ck
    def status(phase,**kw):
        record=dict(phase=phase,utc=time.time(),checkpoint_step=a.step,**kw)
        tmp=job/'status.tmp';tmp.write_text(json.dumps(record,indent=2));tmp.replace(job/'status.json')
    def environment(gpu):
        e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(job/'candidate/src'),
            TORCH_HOME=str(root/'cache/torch'),TORCH_EXTENSIONS_DIR=str(root/'cache/torch_extensions'),
            TMPDIR=str(root/'cache/tmp'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        return e
    def read_rows(folder):
        rows=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
        indexed={(r['stream_id'],r['frame_index']):r for r in rows}
        assert len(indexed)==len(rows),'Duplicate frame'
        return indexed
    def sample_resources(phase):
        p=Path('/sys/fs/cgroup/memory')
        row=dict(utc=time.time(),phase=phase,container_bytes=int((p/'memory.usage_in_bytes').read_text()),
                 container_limit=int((p/'memory.limit_in_bytes').read_text()))
        with (job/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    status('starting',checkpoint_sha256=checkpoint_hash,shared_with_training=True)
    reference_manifests={m:json.loads((reference/m/'manifest.json').read_text()) for m in ('lip','lip_fp')}
    reference_rows={m:read_rows(reference/m) for m in ('lip','lip_fp')}
    assert all(m['completed'] and not m['quick_subset'] and m['full_sequences'] for m in reference_manifests.values())
    assert sha(job/'candidate/tools/eval_lip_fp.py')==reference_manifests['lip']['evaluator_sha256']
    assert sha(job/'config.yaml')==sha(reference/'config.yaml')
    expected=reference_rows['lip'].keys();assert len(expected)==23200
    protocol=('split','mode','initial_pose_source','split_hash','mesh_hash','full_sequences','quick_subset','method','history_state_source','fp_iterations','precision')
    results={}
    try:
        for method in a.methods:
            folder=job/method;folder.mkdir(exist_ok=False);jobs=[]
            for rank in range(8):
                log=(folder/f'rank{rank}.log').open('w')
                cmd=[str(root/'.venv-fp/bin/python'),str(job/'candidate/tools/eval_lip_fp.py'),
                     '--config',str(job/'config.yaml'),'--checkpoint',str(checkpoint),
                     '--data-root',str(root/'cache/raw_full_20260910'),'--index-root','cache/dexycb_s0',
                     '--out',str(folder/f'rank{rank}'),'--method',method,'--rank',str(rank),'--world-size','8']
                p=subprocess.Popen(cmd,env=environment(rank),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                jobs.append((p,log))
            status('evaluating_'+method,pids=[p.pid for p,f in jobs],shared_with_training=True)
            while any(p.poll() is None for p,f in jobs):
                if any(p.poll() not in (None,0) for p,f in jobs):
                    for p,f in jobs:
                        if p.poll() is None:p.terminate()
                    for p,f in jobs:p.wait();f.close()
                    raise RuntimeError('Evaluation worker failed; see rank logs')
                sample_resources(method);time.sleep(5)
            for p,f in jobs:f.close();assert p.returncode==0
            rows=[];manifests=[]
            for rank in range(8):
                d=folder/f'rank{rank}';m=json.loads((d/'manifest.json').read_text())
                assert m['completed'] and m['checkpoint']['global_step']==a.step and m['checkpoint']['sha256']==checkpoint_hash
                assert all(m[k]==reference_manifests[method][k] for k in protocol)
                assert m['fp_refiner_sha256']==reference_manifests[method]['fp_refiner_sha256']
                rows.extend(read_rows(d).values());manifests.append(m)
            ids=sorted(s for m in manifests for s in m['streams'])
            assert ids==reference_manifests[method]['streams'] and len(ids)==len(set(ids))==320
            indexed={(r['stream_id'],r['frame_index']):r for r in rows}
            assert len(indexed)==len(rows)==23200 and indexed.keys()==expected
            assert all(not r['nonfinite_output'] for r in rows)
            assert all(indexed[k][f]==reference_rows[method][k][f] for k in expected for f in ('object_id','visibility','visibility_bin','moving'))
            rows.sort(key=lambda r:(r['stream_id'],r['frame_index']))
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report=summarize(rows);report['latency_scope']='Evaluation shares GPUs with training; no speed comparison with reference'
            (folder/'metrics.json').write_text(json.dumps(report,indent=2));results[method]=report
            manifest=manifests[0].copy();manifest.update(completed=True,frames=len(rows),streams=ids,shard_rank=None,shard_count=8,
                reference_population_verified=True,reference=str(reference/method),shared_with_training=True)
            manifest['first_five_consecutive_adds_failures']={k:v for m in manifests for k,v in m['first_five_consecutive_adds_failures'].items()}
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
        within=dict(matched=True,checkpoint_step=a.step,checkpoint_sha256=checkpoint_hash,frames=23200,streams=320,physical_sequences=40,
                    **{method:results[method]['macro_object'] for method in a.methods})
        if 'lip' in results and 'lip_fp' in results:
            within['lip_fp_minus_lip']={k:results['lip_fp']['macro_object'][k]-v for k,v in results['lip']['macro_object'].items()}
        (job/'comparison.json').write_text(json.dumps(within,indent=2))
        cross=dict(matched=True,early_step=a.step,reference_step=reference_manifests['lip']['checkpoint']['global_step'],
                   frames=23200,streams=320,scope='Full s0 val; same frame population, GT-first-frame initialization and frozen FP refiner; not final test or a training ablation',methods={})
        for method in a.methods:
            later=json.loads((reference/method/'metrics.json').read_text())['macro_object'];early=results[method]['macro_object']
            cross['methods'][method]=dict(early=early,reference=later,reference_minus_early={k:later[k]-v for k,v in early.items()})
        (job/'comparison_to_31000.json').write_text(json.dumps(cross,indent=2))
        status('completed',frames_per_method=23200,streams_per_method=320,checkpoint_sha256=checkpoint_hash)
    except Exception as e:status('failed',error=repr(e));raise

if __name__=='__main__':main()
