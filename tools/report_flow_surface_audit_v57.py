"""Reduce frozen geometry probes, keeping full-domain/fallback coverage explicit."""
import argparse,json,statistics
import numpy as np
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    variants=('baseline','triangle','rigid','metric_recovered','metric_measured','oracle_triangle','oracle_rigid','oracle_cad')
    result={};lines=['# V57 frozen flow / depth / CAD geometry audit','',
        'V56 weights frozen. 64 controlled10-degree frames and32 controlled60-degree frames.',
        'Training-partition physical holdout; no native/pose evaluation. PnP/3D fitting are diagnostics only.',
        'Means are per eligible frame on original target masks; unavailable candidate pixels retain the original prediction.',
        'Oracle branches use labels and must never be interpreted as deployed accuracy.','',
        '| Rotation | Occlusion | Method | Source | Frames | XYZ mm | Depth mm | Candidate coverage |',
        '|---:|---|---|---|---:|---:|---:|---:|']
    for angle in (10,60):
        rows=[]
        for rank in range(8):
            d=a.root/f'angle{angle}'/f'rank{rank}'
            receipt=json.loads((d/'receipt.json').read_text());assert receipt['completed'] and receipt['flow_surface_audit']
            shard=[json.loads(x) for x in (d/'frames.jsonl').read_text().splitlines()]
            assert len(shard)==receipt['records'];rows+=shard
        assert len(rows)==(64 if angle==10 else 32)
        group={}
        for variant in variants:
            group[variant]={}
            for heavy in (False,True):
                for source in ('real','proxy'):
                    values=[(r['metrics'][source] if variant=='baseline' else r['flow_surface'][variant][source]) for r in rows if r['heavy']==heavy]
                    values=[v for v in values if v]
                    reduced=dict(frames=len(values))
                    for key in ('canonical_xyz_mm','depth_mm','coverage'):
                        good=[v[key] for v in values if v.get(key) is not None]
                        reduced[key]=statistics.mean(good) if good else None
                    group[variant][('heavy_' if heavy else 'nonheavy_')+source]=reduced
                    fmt=lambda x:'n/a' if x is None else f'{x:.3f}'
                    lines.append(f'| {angle} | {"heavy" if heavy else "nonheavy"} | {variant} | {source} | {len(values)} | {fmt(reduced["canonical_xyz_mm"])} | {fmt(reduced["depth_mm"])} | {fmt(reduced["coverage"])} |')
            if variant not in ('baseline','triangle','oracle_triangle','oracle_cad'):
                group[variant]['accepted']=sum(r['flow_surface'][variant]['receipt']['accepted'] for r in rows)
        group['observed_evidence']=dict(
            measured_anchors=sum(r['flow_surface']['metric_measured']['receipt']['measured_anchors'] for r in rows),
            mean_measured_fraction=statistics.mean(r['flow_surface']['metric_measured']['receipt']['measured_fraction_mean'] for r in rows),
            point_visible_above_half=sum(r['flow_surface']['point_counts']['point_visible_above_half'] for r in rows),
            measured_depth_available=sum(r['flow_surface']['point_counts']['measured_depth_available'] for r in rows),
            max_point_visibility=max(r['flow_surface']['point_counts']['point_visibility_max'] or 0 for r in rows))
        def visibility(selected):
            score=np.array([v for r in selected for v in r['flow_surface']['visibility_samples']['score']])
            label=np.array([v for r in selected for v in r['flow_surface']['visibility_samples']['label']],dtype=bool)
            pos,neg=score[label],score[~label]
            auc=float(((pos[:,None]>neg[None]).sum()+.5*(pos[:,None]==neg[None]).sum())/(len(pos)*len(neg))) if len(pos) and len(neg) else None
            return dict(points=len(score),positive=int(label.sum()),auroc=auc,
                brier=float(np.mean((score-label)**2)),positive_mean=float(pos.mean()) if len(pos) else None,
                negative_mean=float(neg.mean()) if len(neg) else None,
                recall_at_half=float((pos>=.5).mean()) if len(pos) else None)
        if all('visibility_samples' in r['flow_surface'] for r in rows):
            group['point_visibility']={name:visibility([r for r in rows if name=='all' or r['heavy']==(name=='heavy')]) for name in ('all','heavy','nonheavy')}
        result[str(angle)]=group
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
