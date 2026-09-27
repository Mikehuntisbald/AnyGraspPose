"""Paired recovery and two-way intervention results; never pose accuracy."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root)
    tags=tuple(f'{arm}_{angle}' for angle in (0,10,60) for arm in ('baseline','control','surface'))
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
        for tag in family:
            assert [(r['seed'],r['stream'],r['heavy'],r['natural']) for r in data[tag]]==keys
            for baseline,candidate in zip(data[family[0]],data[tag]):
                for region in ('real','proxy'):
                    a,b=baseline['metrics'][region],candidate['metrics'][region]
                    assert (a is None)==(b is None)
                    if a is not None:
                        assert a['pixels']==b['pixels'] and a['canonical_pixels']==b['canonical_pixels']
                for region in ('observed','real','proxy'):
                    key=f'flow_{region}_count'
                    assert baseline['flow'][key]==candidate['flow'][key]
    result={};lines=['# V68 matched joint geometry: original versus supervised surface feedback','',
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
                for name in ('canonical_xyz_mm','depth_mm','camera_xyz_mm','camera_normal_deg','xyz_within_10mm'):
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
    lines+=['', '## Flow by requested occlusion condition', '',
            '| Variant | Heavy | Source | Frames | Round 1 EPE px |',
            '|---|---|---|---:|---:|']
    for tag,rows in data.items():
        result[tag]['flow_by_occlusion']={}
        for heavy in (False,True):
            for region in ('observed','real','proxy'):
                vals=[r['flow'][f'flow1_{region}_epe'] for r in rows
                      if r['heavy']==heavy and r['flow'][f'flow_{region}_count']>0]
                key=f'{heavy}/{region}'
                value=statistics.mean(vals) if vals else None
                result[tag]['flow_by_occlusion'][key]=dict(frames=len(vals),epe=value)
                if vals:lines.append(f'| {tag} | {heavy} | {region} | {len(vals)} | {value:.3f} |')
    changes={}
    lines+=['', '## Paired candidate minus control geometry error', '',
            'Negative error change is better. Win fraction counts strictly lower error per eligible frame.', '',
            '| Angle | Heavy | Source | Metric | Frames | Mean change | Win fraction |',
            '|---:|---|---|---|---:|---:|---:|']
    for angle in (0,10,60):
        for heavy in (False,True):
            for region in ('real','proxy'):
                for metric in ('canonical_xyz_mm','depth_mm'):
                    deltas=[]
                    for control,surface in zip(data[f'control_{angle}'],data[f'surface_{angle}']):
                        if control['heavy']!=heavy:continue
                        a,b=control['metrics'][region],surface['metrics'][region]
                        if a and a.get(metric) is not None and b.get(metric) is not None:
                            deltas.append(b[metric]-a[metric])
                    if not deltas:continue
                    mean=statistics.mean(deltas);wins=sum(x<0 for x in deltas)/len(deltas)
                    changes[f'{angle}/{heavy}/{region}/{metric}']=dict(frames=len(deltas),mean_delta=mean,win_fraction=wins)
                    lines.append(f'| {angle} | {heavy} | {region} | {metric} | {len(deltas)} | {mean:.4f} | {wins:.1%} |')
    result['paired_candidate_minus_control']=changes
    (root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
