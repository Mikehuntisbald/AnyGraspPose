"""Build a verified engineering handoff from completed, retained experiment artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import source_hash,sha
from lip.engine.stream_config import load_stream_config,config_hash
from lip.engine.stream_state import CACHE_CONTRACT


def historical_hashes(project):
    """Reconstruct the earlier source digests by reversing the audited edits.

    This binds preflight trials to the exact source, rather than silently replacing
    an old approval digest with the final one. The final CPU suite must also pass.
    """
    root=project/'src/lip';files={str(p.relative_to(root)):p.read_text() for p in sorted(root.rglob('*.py'))}
    def replace_file(name,before,after):
        if before not in files[name]:raise ValueError('Source audit pattern absent: '+name)
        files[name]=files[name].replace(before,after,1)
    def digest():
        h=hashlib.sha256()
        for name,text in sorted(files.items()):h.update(name.encode());h.update(text.encode())
        return h.hexdigest()
    replace_file('train_stream.py',"    if a.max_steps is not None and a.max_steps<1:p.error('--max-steps must be positive; zero must never expand into a long run')\n",'')
    replace_file('engine/stream_state.py',"        if self.frame_id.ndim!=1 or self.timestamp.shape!=self.frame_id.shape or self.stream_tag.shape!=self.frame_id.shape:\n            raise ValueError('Timestamp, frame ID and stream tag must each have shape [B]')\n",'')
    replace_file('evaluation/metrics.py',"def errors(pred, gt, points, d, dtype='f8'):\n    \"\"\"Legacy float64 by default; streaming requests FP32 pose/error arithmetic.\n\n    SciPy's nearest-neighbor search uses double internally; its distances are\n    narrowed before the requested-precision ADD-S reduction.\n    \"\"\"\n    pred,gt,points=[np.asarray(x,dtype=dtype) for x in (pred,gt,points)]", "def errors(pred, gt, points, d):\n    pred,gt,points=[np.asarray(x,dtype='f8') for x in (pred,gt,points)]")
    replace_file('evaluation/metrics.py',"cKDTree(b).query(a)[0].astype(dtype,copy=False).mean()","cKDTree(b).query(a)[0].mean()")
    replace_file('evaluate_stream.py',"    assert m['streams']==m['expected_streams'] and len(rows)==m['expected_frames']\n    m['population_verified']=True\n",'')
    replace_file('evaluate_stream.py',"    expected_streams=[s['stream_id'] for s in streams]\n    expected_frames=sum(min(s['num_frames'],a.max_frames) if a.max_frames is not None else s['num_frames'] for s in streams)\n",'')
    replace_file('evaluate_stream.py',"streams=[s['stream_id'] for s in streams],\n        expected_streams=expected_streams,expected_frames=expected_frames,metric_pose_precision='fp32; SciPy nearest-neighbor search internal float64')", "streams=[s['stream_id'] for s in streams])")
    replace_file('evaluate_stream.py',"float(mesh['diameter']),dtype='f4')","float(mesh['diameter']))")
    dual=digest()
    replace_file('engine/stream_config.py',"        partial_new=model.migration_status.get(name,{}).get('partial_new',False)\n        category='rgb' if name.startswith('rgb.') else ('loaded' if status in ('loaded','remapped') and not partial_new else 'new')", "        category='rgb' if name.startswith('rgb.') else ('loaded' if status in ('loaded','remapped') else 'new')")
    replace_file('engine/stream_checkpoint.py',"        copied=(tensor.numel()//2 if entry.get('partial_new') else tensor.numel()) if entry['status'] in ('loaded','remapped') else 0\n        report[name]=dict(**entry,numel=tensor.numel(),loaded_numel=copied,initialized_numel=tensor.numel()-copied,shape=list(tensor.shape))", "        report[name]=dict(**entry,numel=tensor.numel(),shape=list(tensor.shape))")
    replace_file('engine/stream_checkpoint.py',"    loaded=sum(v['loaded_numel'] for k,v in report.items() if k in param_names)","    loaded=sum(v['numel'] for k,v in report.items() if k in param_names and v['status'] in ('loaded','remapped'))")
    return dict(single=digest(),dual=dual)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True);p.add_argument('--init-root',required=True);a=p.parse_args()
    root=Path(a.out).resolve();project=Path(__file__).resolve().parents[1]
    read=lambda path:json.loads(Path(path).read_text())
    tests=ET.parse(root/'delivery_tests.xml').findall('.//testsuite')
    assert tests and all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in tests)
    count=sum(int(s.get('tests',0))-int(s.get('skipped',0)) for s in tests)
    assert count>=70
    historical=historical_hashes(project)
    result=dict(implementation_passed=True,short_training_passed=True,converged_accuracy_validated=False,
        tests_passed=count,foundationpose_test='deselected: legacy test is an explicit skipped fixture placeholder',
        source_sha256=source_hash(),cache_contract=CACHE_CONTRACT,architectures={},historical_source_hashes=historical)
    for name in ('single','dual'):
        folder=root/name;approval=read(folder/'approval.json')
        original=approval.get('preflight_source_sha256',approval['source_sha256'])
        assert historical[name]==original,(name,historical[name],original)
        c=load_stream_config(project/f'configs/stream_lip_v2_{name}.yaml')
        assert approval['approved'] and approval['config_hash']==config_hash(c)
        last=torch.load(folder/'ddp/last.pt',map_location='cpu',weights_only=False)
        assert last['new_stage_step']==last['scheduler']['last_epoch']==53
        assert last['sampler_position']==3392 and len(last['rng'])==8
        assert last['config_hash']==config_hash(c) and last['cache_contract']==CACHE_CONTRACT
        resumes=[read(folder/f'ddp/resume_rank{r}.json') for r in range(8)]
        assert all(r['passed'] and r['loaded_step']==50 and r['sampler_position']==3200 for r in resumes)
        logs=[list(map(json.loads,(folder/f'ddp/rank{r}.jsonl').read_text().splitlines())) for r in range(8)]
        assert all(len(log)==53 and all(r['actual_supervised_frames_rank']==128 and r['actual_supervised_frames_global']==1024 and r['kv_has_training_graph'] for r in log) for log in logs)
        before=read(folder/'overfit_before.json');after=read(folder/'overfit_after.json')
        means=lambda report:np.mean([r['metrics'] for r in report['rows']],axis=0).tolist()
        memory=list(map(json.loads,(folder/'memory_probe/rank0.jsonl').read_text().splitlines()))
        migration=read(Path(a.init_root)/('dual/weight_migration.json' if name=='dual' else 'weight_migration.json'))
        result['architectures'][name]=dict(migration_coverage=migration['coverage'],parent=migration['parent'],
            overfit=dict(steps=300,fragments=16,before_loss=before['mean_loss'],after_loss=after['mean_loss'],before_metrics=means(before),after_metrics=means(after)),
            memory_probe_peak_bytes=max(r['peak_allocated'] for r in memory),
            ddp=dict(steps=50,resume_steps=3,mean_step_seconds=float(np.mean([r['seconds'] for log in logs for r in log[5:50]])),
                max_peak_bytes=max(r['peak_allocated'] for log in logs for r in log),supervised_frames_per_step=1024),
            depth_issue=read(folder/'geometry.json'))
        approval.update(preflight_source_sha256=original,source_sha256=source_hash(),
            source_revalidation=dict(historical_digest_reconstructed=True,final_tests_sha256=sha(root/'delivery_tests.xml'),
                scope='Final source adds FP32 evaluation and strict population checks, metadata/CLI guards; dual partial-query migration accounting/LR was present in dual trials and does not change single groups.'),
            final_tests_passed=count)
        (folder/'approval.json').write_text(json.dumps(approval,indent=2));del last
    eager=read(root/'benchmark_eager_all/benchmark.json');compiled=read(root/'benchmark_compiled/benchmark.json')
    for report in (eager,compiled):
        assert report['completed'] and report['batch_size']==1
        for name in ('stream_single','stream_dual'):
            for scope in ('core','end_to_end'):
                r=report['configurations'][name][scope]
                assert r['encoder_images_per_update']==dict(rgb=1.,geometry=1.) and r['render_calls_per_update']==1
                assert r['cache_bytes']==557056 and r['fp_calls']==r['critic_calls']==0
    result['benchmarks']=dict(eager=eager,compiled=compiled)
    status=read(root/'validation/status.json');assert status['phase']=='completed'
    result['validation']=read(root/'validation/comparison.json')
    result['validation']['interpretation']='These are 300-step fixed-fragment checkpoints, with different warm-start lineage; accuracy is far below legacy and does not justify production replacement.'
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    labels=['V1 34700','stream single','stream dual'];names=['legacy','stream_single','stream_dual'];x=np.arange(3)
    axes[0].bar(x,[result['validation']['results'][n]['excluding_init']['adds_01']*100 for n in names],color=['#777777','#4477aa','#cc6677'])
    axes[0].set_xticks(x,labels);axes[0].set_ylim(0,100);axes[0].set_ylabel('ADD-S@0.1d (%)');axes[0].set_title('23,200-frame validation, eager\nInitialization excluded; short-trained v2')
    for offset,report,label in [(-.18,eager,'Eager'),(.18,compiled,'Compiled encoder (v2 only)')]:
        axes[1].bar(x+offset,[report['configurations'][n]['end_to_end']['latency_ms']['p95'] for n in names],width=.36,label=label)
    axes[1].axhline(33.333,color='k',linestyle='--',linewidth=.7);axes[1].set_xticks(x,labels);axes[1].set_ylabel('E2E P95 latency (ms)');axes[1].legend(fontsize=8);axes[1].set_title('One H20, batch=1, 120 timed updates\nCompiled accuracy not validated')
    for name,color in [('single','#4477aa'),('dual','#cc6677')]:
        rows=list(map(json.loads,(root/name/'overfit/rank0.jsonl').read_text().splitlines()))
        loss=np.array([r['loss'] for r in rows]);smooth=np.convolve(loss,np.ones(16)/16,mode='valid')
        axes[2].plot(np.arange(16,len(loss)+1),smooth,label=name,color=color)
    axes[2].set_xlabel('Short-training stage step');axes[2].set_ylabel('16-step rolling training loss');axes[2].legend();axes[2].set_title('16 real continuous fragments\nDifferent warm-start lineages')
    fig.savefig(root/'summary.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(implementation_passed=True,short_training_passed=True,tests_passed=count,converged_accuracy_validated=False,source_sha256=source_hash()),indent=2))

if __name__=='__main__':main()
