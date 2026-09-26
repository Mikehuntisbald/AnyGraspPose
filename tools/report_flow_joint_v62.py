"""Paired recovery and two-way intervention results; never pose accuracy."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root)
    tags=tuple(f'{arm}_{angle}' for angle in (0,10,60) for arm in ('baseline','control','strong'))
    data={}
    for tag in tags:
        rows=[]
        for rank in range(8):
            p=root/'probe'/tag/f'rank{rank}'
            receipt=json.loads((p/'receipt.json').read_text())
            assert receipt['completed'] and receipt['physical_holdout'] and receipt['training_split_only']
            shard=[json.loads(x) for x in (p/'frames.jsonl').read_text().splitlines()]
            assert len(shard)==receipt['records'];rows+=shard
        data[tag]=rows
    for family in (tags[:3],tags[3:6],tags[6:]):
        keys=[(r['seed'],r['stream'],r['heavy'],r['natural']) for r in data[family[0]]]
        for tag in family:assert [(r['seed'],r['stream'],r['heavy'],r['natural']) for r in data[tag]]==keys
    result={};lines=['# V62 matched joint geometry: write gain1 versus100','',
        'Training-partition physical holdout; 32 paired observations at0 degrees,64 at10 degrees,32 at60 degrees.',
        'GT-perturbed references are a controlled diagnostic, not native rollout or pose accuracy.',
        'Means below are per eligible frame, with original unfiltered real/proxy masks. No confidence filtering.', '',
        '| Variant | Occlusion | Source | Frames | Canonical XYZ mm | Depth mm |',
        '|---|---|---|---:|---:|---:|']
    for tag,rows in data.items():
        result[tag]={}
        for heavy in (False,True):
            for source in ('real','proxy'):
                values=[r['metrics'][source] for r in rows if r['heavy']==heavy and r['metrics'][source]]
                key=('heavy_' if heavy else 'nonheavy_')+source
                row=dict(frames=len(values))
                for name in ('canonical_xyz_mm','depth_mm'):
                    numbers=[v[name] for v in values if v.get(name) is not None]
                    row[name]=statistics.mean(numbers) if numbers else None
                result[tag][key]=row
                fmt=lambda v:'n/a' if v is None else f'{v:.3f}'
                lines.append(f'| {tag} | {"heavy" if heavy else "nonheavy"} | {source} | {row["frames"]} | {fmt(row["canonical_xyz_mm"])} | {fmt(row["depth_mm"])} |')
        flow={}
        for stage in (0,1):
            for source in ('observed','real','proxy'):
                values=[r['flow'][f'flow{stage}_{source}_epe'] for r in rows if 'flow' in r and r['flow'][f'flow_{source}_count']>0]
                flow[f'round{stage}_{source}_epe']=statistics.mean(values) if values else None
        result[tag]['flow']=flow
    lines+=['','## Flow endpoint errors in crop pixels','', '| Variant | Source | Round 0 | Round 1 |','|---|---|---:|---:|']
    for tag,values in result.items():
        for source in ('observed','real','proxy'):
            a,b=[values['flow'][f'round{i}_{source}_epe'] for i in (0,1)]
            if a is not None:lines.append(f'| {tag} | {source} | {a:.3f} | {b:.3f} |')
    (root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
