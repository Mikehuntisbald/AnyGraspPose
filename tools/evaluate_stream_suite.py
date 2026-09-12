"""Eight independent full-sequence validation shards, then strict population checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha
from lip.evaluation.metrics import summarize


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--single-checkpoint',required=True);p.add_argument('--dual-checkpoint',required=True)
    p.add_argument('--legacy-checkpoint');p.add_argument('--legacy-config');p.add_argument('--data-root',required=True)
    p.add_argument('--index-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--single-config',default='configs/stream_lip_v2_single.yaml');p.add_argument('--dual-config',default='configs/stream_lip_v2_dual.yaml');a=p.parse_args()
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
    if occupied:raise RuntimeError('GPU jobs already running; suite does not preempt: '+occupied)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);project=Path(__file__).resolve().parents[1]
    source_sha={str(p.relative_to(project)):sha(p) for p in sorted((project/'src/lip').rglob('*.py'))}
    variants=[('stream_single',a.single_checkpoint,a.single_config),('stream_dual',a.dual_checkpoint,a.dual_config)]
    if a.legacy_checkpoint:
        if not a.legacy_config:p.error('--legacy-config is required with a legacy checkpoint')
        variants.insert(0,('legacy',a.legacy_checkpoint,a.legacy_config))
    keys_reference=None;summary={}
    for name,checkpoint,config in variants:
        folder=out/name;folder.mkdir();jobs=[]
        for rank in range(8):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),PYTHONPATH=str(project/'src'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
            command=[sys.executable]
            command+=['tools/eval_lip_fp.py','--method','lip'] if name=='legacy' else ['-m','lip.evaluate_stream']
            command+=['--config',str(Path(config).resolve()),'--checkpoint',str(Path(checkpoint).resolve()),'--data-root',str(Path(a.data_root).resolve()),
                '--index-root',str(Path(a.index_root).resolve()),'--out',str(folder/f'rank{rank}'),'--rank',str(rank),'--world-size','8']
            log=(folder/f'rank{rank}.log').open('w');proc=subprocess.Popen(command,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((proc,log,command))
        while any(proc.poll() is None for proc,log,cmd in jobs):
            if any(proc.poll() not in (None,0) for proc,log,cmd in jobs):
                # Only stop this tool's failed evaluation siblings, never external jobs.
                for proc,log,cmd in jobs:
                    if proc.poll() is None:proc.terminate()
                for proc,log,cmd in jobs:proc.wait();log.close()
                (out/'status.json').write_text(json.dumps(dict(phase='failed',variant=name)))
                raise RuntimeError('Evaluation shard failed; see rank logs')
            (out/'status.json').write_text(json.dumps(dict(phase='evaluating',variant=name,pids=[proc.pid for proc,log,cmd in jobs],utc=time.time())))
            time.sleep(3)
        for proc,log,cmd in jobs:log.close();assert proc.returncode==0
        if name!='legacy':
            from lip.evaluate_stream import merge_shards
            merge_shards(folder,8)
        else:
            rows=[];manifests=[]
            for rank in range(8):
                d=folder/f'rank{rank}';manifests.append(json.loads((d/'manifest.json').read_text()))
                rows.extend(map(json.loads,(d/'predictions.jsonl').read_text().splitlines()))
            assert all(m['completed'] and m['checkpoint']['sha256']==sha(checkpoint) for m in manifests)
            rows.sort(key=lambda r:(r['stream_id'],r['frame_index']));(folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report=summarize(rows);tracked=[r for r in rows if r['frame_index']!=0];report['excluding_initialization']=summarize(tracked)
            hard=[r for r in tracked if r['moving'] and r['visibility'] is not None and r['visibility']<.3]
            report['moving_and_visibility_lt_03']=summarize(hard) if hard else dict(count=0)
            (folder/'metrics.json').write_text(json.dumps(report,indent=2))
            m=manifests[0];m.update(frames=len(rows),streams=sorted(s for x in manifests for s in x['streams']),shard_rank=None)
            (folder/'manifest.json').write_text(json.dumps(m,indent=2))
        rows=list(map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()))
        keys={(r['stream_id'],r['frame_index']) for r in rows}
        assert len(keys)==len(rows)==23200 and len(set(r['stream_id'] for r in rows))==320
        if keys_reference is None:keys_reference=keys
        assert keys==keys_reference,'Different frame population'
        report=json.loads((folder/'metrics.json').read_text())
        summary[name]=dict(checkpoint_sha256=sha(checkpoint),including_init=report['macro_object'],
            excluding_init=report['excluding_initialization']['macro_object'],
            moving_visibility_lt_03=report['moving_and_visibility_lt_03'])
        (out/'comparison.json').write_text(json.dumps(dict(scope='Full s0 validation after bounded short training, not a converged-model claim or matched-budget architecture ablation. Dual starts from single short-trained weights.',
            frames=23200,streams=320,results=summary,source_sha256=source_sha),indent=2))
    (out/'status.json').write_text(json.dumps(dict(phase='completed',frames_per_variant=23200,variants=list(summary)),indent=2))

if __name__=='__main__':main()
