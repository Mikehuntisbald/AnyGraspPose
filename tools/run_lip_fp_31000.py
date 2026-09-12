"""Wait for the requested checkpoint pause, evaluate both loops, and stay paused."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from lip.evaluation.metrics import summarize

R=Path('/mnt/why/dexycb_lip');os.chdir(R);J=R/'runs/lip_fp_31000'

def status(phase,**kw):
    (J/'status.json').write_text(json.dumps(dict(phase=phase,utc=time.time(),**kw),indent=2))

def env(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(J/'candidate/src'),
        TORCH_HOME=str(R/'cache/torch'),TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),
        TMPDIR=str(R/'cache/tmp'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    return e

def evaluate(method,smoke=False):
    folder=J/('smoke' if smoke else method);folder.mkdir(exist_ok=True)
    jobs=[]
    for rank in range(1 if smoke else 8):
        output=folder/f'rank{rank}';log=(folder/f'rank{rank}.log').open('w')
        cmd=[str(R/'.venv-fp/bin/python'),str(J/'candidate/tools/eval_lip_fp.py'),
             '--config',str(J/'config.yaml'),'--checkpoint',str(J/'resume_31000.pt'),
             '--data-root',str(R/'cache/raw_full_20260910'),'--out',str(output),
             '--method',method,'--rank',str(rank),'--world-size',str(1 if smoke else 8)]
        if smoke:cmd+=['--limit-streams','2','--max-frames','8']
        process=subprocess.Popen(cmd,env=env(rank),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append((process,log))
    status('smoke' if smoke else 'evaluating_'+method,pids=[p.pid for p,f in jobs])
    while any(p.poll() is None for p,f in jobs):
        if any(p.poll() not in (None,0) for p,f in jobs):
            for p,f in jobs:
                if p.poll() is None:p.terminate()
            raise RuntimeError('Evaluation worker failed')
        time.sleep(5)
    for p,f in jobs:f.close();assert p.returncode==0
    if smoke:
        m=json.loads((folder/'rank0/manifest.json').read_text())
        rr=[json.loads(x) for x in (folder/'rank0/predictions.jsonl').read_text().splitlines()]
        assert m['completed'] and len(rr)==16 and all(not r['nonfinite_output'] for r in rr)
        assert sum('lip_proposal_centered' in r for r in rr)==14
        (J/'smoke_passed.json').write_text(json.dumps(dict(passed=True,streams=2,frames=16,actual_frozen_FP_calls=14),indent=2))
        return
    rows=[];manifests=[]
    for rank in range(8):
        p=folder/f'rank{rank}';m=json.loads((p/'manifest.json').read_text());assert m['completed']
        manifests.append(m);rows.extend(json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines())
    streams=[s for s in map(json.loads,(R/'cache/dexycb_s0/streams.jsonl').read_text().splitlines()) if s['split']=='val']
    ids=sorted(s['stream_id'] for s in streams)
    assert len(ids)==320 and sorted(s for m in manifests for s in m['streams'])==ids
    expected=set()
    for s in streams:
        with np.load(R/'cache/dexycb_s0'/s['pose_cache']) as z:expected.update((s['stream_id'],int(f)) for f in z['frames'])
    assert len(rows)==len(expected)==23200 and {(r['stream_id'],r['frame_index']) for r in rows}==expected
    assert all(not r['nonfinite_output'] for r in rows)
    for m in manifests:
        assert m['checkpoint']['sha256']==manifests[0]['checkpoint']['sha256'] and m['checkpoint']['global_step']==31000
        assert m['split_hash']==manifests[0]['split_hash'] and m['mesh_hash']==manifests[0]['mesh_hash']
    rows.sort(key=lambda r:(r['stream_id'],r['frame_index']))
    (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=summarize(rows);report['latency_scope']='8 independent batch-one workers; no isolated serial FPS claim'
    (folder/'metrics.json').write_text(json.dumps(report,indent=2))
    manifest=manifests[0].copy();manifest.update(streams=ids,frames=len(rows),completed=True,shard_rank=None,shard_count=8)
    manifest['first_five_consecutive_adds_failures']={k:v for m in manifests for k,v in m['first_five_consecutive_adds_failures'].items()}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))

def main():
    status('waiting_checkpoint_31000')
    while not (J/'paused.json').exists():time.sleep(5)
    pause=json.loads((J/'paused.json').read_text());assert pause['step']==31000
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(5)
    tests=(J/'tests.log').read_text();assert 'passed' in tests and 'failed' not in tests and 'ERROR' not in tests
    evaluate('lip_fp',smoke=True)
    evaluate('lip')
    evaluate('lip_fp')
    a,b=[json.loads((J/m/'metrics.json').read_text()) for m in ('lip','lip_fp')]
    rows=[]
    for m in ('lip','lip_fp'):
        rows.append({(r['stream_id'],r['frame_index']):r for r in map(json.loads,(J/m/'predictions.jsonl').read_text().splitlines())})
    assert rows[0].keys()==rows[1].keys()
    assert all(rows[0][k][f]==rows[1][k][f] for k in rows[0] for f in ('object_id','visibility','visibility_bin','moving'))
    comparison=dict(matched=True,checkpoint_step=31000,frames=23200,streams=320,physical_sequences=40,
        lip=a['macro_object'],lip_fp=b['macro_object'],lip_fp_minus_lip={k:b['macro_object'][k]-a['macro_object'][k] for k in a['macro_object']},
        scope='Full official s0 val; same checkpoint and frames; GT first frame only; not final test')
    (J/'comparison.json').write_text(json.dumps(comparison,indent=2))
    status('evaluation_complete_training_paused',comparison=comparison)

if __name__=='__main__':
    try:main()
    except Exception as e:status('failed_training_remains_paused',error=repr(e));raise
