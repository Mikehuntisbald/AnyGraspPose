"""Paired label-contract report; never relabel historical raw XYZ as CAD XYZ."""
import argparse,json,statistics
from pathlib import Path
import torch


def mean_by_sequence(rows,extract):
    values={}
    for row in rows:
        metric=extract(row)
        if metric is None:continue
        physical='/'.join(row['stream'].split('|')[0].split('/')[:2])
        for key,value in metric.items():
            if isinstance(value,(int,float)):values.setdefault(key,{}).setdefault(physical,[]).append(value)
    return {k:statistics.mean(statistics.mean(xs) for xs in groups.values()) for k,groups in values.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    assert json.loads((root/'status.json').read_text())['completed']
    starts={arm:torch.load(root/arm/'runs/seed42/initial.pt',map_location='cpu',weights_only=False) for arm in ('raw','canonical')}
    assert starts['raw']['model'].keys()==starts['canonical']['model'].keys()
    assert all(torch.equal(value,starts['canonical']['model'][name]) for name,value in starts['raw']['model'].items())
    assert starts['raw']['optimizer']==starts['canonical']['optimizer']
    pose_frozen={}
    for arm in starts:
        terminal=torch.load(root/arm/'runs/seed42/last.pt',map_location='cpu',weights_only=False)
        assert terminal['step']==200
        names=[n for n in terminal['model'] if n=='query' or n.startswith(('head.','object_attn.','object_norm.','geometry_readout.','core.feature_'))]
        assert all(torch.equal(terminal['model'][n],starts[arm]['model'][n]) for n in names)
        assert json.loads((root/arm/'runs/seed42/resume2.json').read_text())['complete_state_verified']
        pose_frozen[arm]=len(names)
        del terminal
    del starts
    rows={};tables={}
    for arm in ('raw','canonical'):
        for step in (0,200):
            name=arm+str(step);data=[]
            for rank in range(8):
                path=root/arm/'probe'/f'step{step}'/f'rank{rank}'
                assert json.loads((path/'receipt.json').read_text())['completed']
                data += [json.loads(s) for s in (path/'frames.jsonl').read_text().splitlines()]
            rows[name]={r['seed']:r for r in data}
            tables[name]={group:{kind:mean_by_sequence([r for r in data if group=='all' or r[group]],lambda r:r['metrics'][kind]) for kind in ('real','proxy')} for group in ('all','heavy','natural')}
    reference=rows['raw0']
    for name,data in rows.items():
        assert data.keys()==reference.keys()
        for seed,row in data.items():
            old=reference[seed]
            assert all(row[k]==old[k] for k in ('stream','heavy','natural','window'))
            for kind in ('real','proxy'):
                x,y=row['metrics'][kind],old['metrics'][kind]
                assert (x is None)==(y is None)
                if x:assert x['pixels']==y['pixels'] and x['canonical_pixels']==y['canonical_pixels']
    report=dict(completed=True,paired_start_model_optimizer_verified=True,paired_records=len(reference),pose_and_feature_head_frozen_tensors=pose_frozen,
                training_partition_only=True,reduction='Equal physical-sequence mass; heavy records within each sequence; empty regions excluded',
                metrics=tables,goal_complete=False,default_model_changed=False)
    (root/'outcome.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# V38 paired surface-identity trial','','Raw sensor-derived XYZ and CAD-canonical XYZ are different targets; real depth is unchanged.','',
           '|Arm|Real raw XYZ mm|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|','|---|---:|---:|---:|---:|---:|']
    for name,t in tables.items():
        r,s=t['heavy']['real'],t['heavy']['proxy']
        lines.append(f"|{name}|{r['xyz_mm']:.3f}|{r['canonical_xyz_mm']:.3f}|{r['depth_mm']:.3f}|{s['xyz_mm']:.3f}|{s['depth_mm']:.3f}|")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
