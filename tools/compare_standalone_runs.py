"""Compare frozen models under the same complete non-GT, zero-FP input protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('selected','baseline','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();runs=dict(selected=a.selected,baseline=a.baseline);protocols={};scores={};bindings={}
    for name,folder in runs.items():
        assert json.loads((folder/'status.json').read_text())['phase']=='completed'
        protocols[name]=json.loads((folder/'protocol.json').read_text())
        scores[name]=json.loads((folder/'occlusion_ar.json').read_text())
        verified=json.loads((folder/'inference_verified.json').read_text())
        assert verified['population_verified'] and scores[name]['completed']
        assert sha(folder/'csv/lip_temporal.csv')==verified['csv_sha256']['lip_temporal']
        bindings[name]={f:sha(folder/f) for f in ('protocol.json','occlusion_ar.json','csv/lip_temporal.csv','inference_verified.json')}
    for key in ('source_sha256','initializer_sha256','targets_sha256','data_root','index_root','fps','methods',
                'history','fp_calls','population','initialization_policy','frame_policy','failure_policy','tools_sha256'):
        assert protocols['selected'][key]==protocols['baseline'][key],key
    assert protocols['selected']['methods']==['lip_temporal'] and protocols['selected']['fp_calls']==0
    assert scores['selected']['physical_sequences']==scores['baseline']['physical_sequences']
    assert scores['selected']['per_cluster_counts']==scores['baseline']['per_cluster_counts']
    count=len(scores['selected']['physical_sequences']);draws=np.random.default_rng(20260913).integers(0,count,(1000,count))
    populations={}
    for bucket,counts in scores['selected']['per_cluster_counts'].items():
        n=np.asarray(counts);values={name:np.asarray(s['per_cluster_successes']['lip_temporal'][bucket]) for name,s in scores.items()}
        if not n.sum():populations[bucket]=dict(targets=0,selected=None,baseline=None,delta_pp=None,ci95_pp=None);continue
        delta=values['selected']-values['baseline'];denominator=n[draws].sum(1);valid=denominator>0
        sampled=delta[draws].sum(1)[valid]/denominator[valid]*100
        populations[bucket]=dict(targets=int(n.sum()),**{name:float(v.sum()/n.sum()*100) for name,v in values.items()},
            delta_pp=float(delta.sum()/n.sum()*100),ci95_pp=np.quantile(sampled,[.025,.975]).tolist(),valid_resamples=int(valid.sum()))
    a.out.mkdir(parents=True,exist_ok=False)
    report=dict(completed=True,protocol='Same direct PoseCNN first legal initializer, full causal RGB-D, same official target denominator and zero FP.',
        checkpoints={name:p['checkpoint_sha256'] for name,p in protocols.items()},bindings=bindings,populations=populations,
        bootstrap='1000 paired physical-sequence resamples, seed 20260913. Descriptive intervals, no multiplicity adjustment.',
        limitations='Architecture and training history differ. This is a matched inference comparison to the fixed internal residual-1000 baseline, not proof of external SOTA or a training-budget-matched architectural effect. Controlled retraining results are reported separately. Single-image leaderboard and bidirectional visibility-aware AUC use different conditions.')
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Frozen standalone LIP: official s0 test comparison','',report['protocol'],'',
        '| Official population | Targets | Selected AR (%) | Residual-1000 AR (%) | Difference (pp), paired 95% CI |',
        '|---|---:|---:|---:|---:|']
    for bucket,v in populations.items():
        if v['targets']:
            lo,hi=v['ci95_pp'];lines.append(f"| {bucket} | {v['targets']} | {v['selected']:.4f} | {v['baseline']:.4f} | {v['delta_pp']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    lines+=['',report['bootstrap'],'',report['limitations']]
    (a.out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
