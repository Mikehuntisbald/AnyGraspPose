"""Matched V40 training receipt and both input-availability probe conditions."""
import argparse,json,statistics
from pathlib import Path
import numpy as np
import torch


def exact(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    initial={arm:torch.load(root/arm/'runs/seed42/initial.pt',map_location='cpu',weights_only=False) for arm in ('corrupted','clean')}
    identity={k:exact(initial['corrupted'][k],initial['clean'][k]) for k in ('model','optimizer','scheduler','rng','sampler_position')}
    assert all(identity.values())
    frozen={}
    for arm in initial:
        assert json.loads((root/arm/'status.json').read_text())['completed']
        assert json.loads((root/arm/'runs/seed42/resume2.json').read_text())['complete_state_verified']
        for rank in range(8):assert json.loads((root/arm/'runs/seed42'/f'paired_input_rank{rank}.json').read_text())['identical_rgbd']
        terminal=torch.load(root/arm/'runs/seed42/last.pt',map_location='cpu',weights_only=False)
        assert terminal['step']==100
        names=[n for n in terminal['model'] if n=='query' or n.startswith(('head.','object_attn.','object_norm.','geometry_readout.','core.feature_'))]
        assert all(torch.equal(terminal['model'][n],initial[arm]['model'][n]) for n in names)
        frozen[arm]=len(names);del terminal
    del initial
    data={};metrics={}
    for arm in ('corrupted','clean'):
        for step in (0,100):
            for setting in ('corrupted','clean'):
                label=f'{arm}_{step}_{setting}';folder=root/arm/'probe'/f"step{step}{'_clean' if setting=='clean' else ''}"
                rows=[]
                for rank in range(8):
                    path=folder/f'rank{rank}';receipt=json.loads((path/'receipt.json').read_text())
                    assert receipt['completed'] and receipt['clean_control']==(setting=='clean')
                    rows += [json.loads(s) for s in (path/'frames.jsonl').read_text().splitlines()]
                data[label]={r['seed']:r for r in rows};metrics[label]={}
                for group in ('all','heavy','natural'):
                    metrics[label][group]={}
                    for kind in ('real','proxy'):
                        by_metric={};count=0
                        for row in rows:
                            value=row['metrics'][kind]
                            if (group!='all' and not row[group]) or value is None:continue
                            count+=1;physical='/'.join(row['stream'].split('|')[0].split('/')[:2])
                            for k,v in value.items():
                                if isinstance(v,(int,float)):by_metric.setdefault(k,{}).setdefault(physical,[]).append(v)
                        result={k:statistics.mean(statistics.mean(v) for v in seq.values()) for k,seq in by_metric.items()}
                        result['eligible_cases']=count;metrics[label][group][kind]=result
    reference=data['corrupted_0_corrupted']
    for label,rows in data.items():
        assert rows.keys()==reference.keys()
        for seed,row in rows.items():
            other=reference[seed]
            assert all(row[k]==other[k] for k in ('stream','heavy','natural','window'))
            for kind in ('real','proxy'):
                x,y=row['metrics'][kind],other['metrics'][kind]
                assert (x is None)==(y is None)
                if x:assert x['pixels']==y['pixels'] and x['canonical_pixels']==y['canonical_pixels']
    result=dict(short_trial_complete=True,paired_start_identity=identity,paired_rgbd_verified_all_ranks=True,pose_and_feature_frozen_tensors=frozen,
                records_per_probe=len(reference),updates_per_arm=100,distinct_observations_per_update=32,pose_hypotheses_per_update=64,
                reduction='Equal physical-sequence mass, then eligible records per sequence',metrics=metrics,
                clean_visibility_fix='Original two-step clean startup preserved in clean_aborted_visibility; restarted from unchanged parent with actual-input visibility labels',
                goal_complete=False,default_model_changed=False,full_validation_required=True)
    (root/'short_outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# V40 paired-estimate100-update trial','','|Training arm/step|Probe input|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|','|---|---|---:|---:|---:|---:|']
    for arm,step in [('corrupted',0),('corrupted',100),('clean',100)]:
        for setting in ('corrupted','clean'):
            t=metrics[f'{arm}_{step}_{setting}']['heavy'];r,s=t['real'],t['proxy']
            lines.append(f"|{arm}/{step}|{setting}|{r['canonical_xyz_mm']:.3f}|{r['depth_mm']:.3f}|{s['canonical_xyz_mm']:.3f}|{s['depth_mm']:.3f}|")
    lines+=['','Clean inputs restore available sensor evidence; those rows are not heavy-occlusion deployment performance. Fixed40 validation is separate and required before extending this curriculum.']
    (root/'SHORT_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
