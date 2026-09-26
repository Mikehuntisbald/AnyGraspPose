"""Reduce frozen write/oracle interventions without filtering geometry targets."""
import argparse,json,statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=[]
    for rank in range(8):
        folder=a.root/f'rank{rank}';receipt=json.loads((folder/'receipt.json').read_text())
        assert receipt['completed'] and receipt['normal_rewrite_bitwise_verified']
        rows.extend(json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines())
    assert len(rows)==64
    result={};lines=['# V61 flow write dependence','',
        '64 frozen V60-extra observations, controlled10/60 degree references, training-partition physical holdout.',
        'Oracle variants replace only known supported front-surface endpoints; other predictions are unchanged. These are diagnostics, not deployable models.',
        'Heavy is the requested augmentation condition. Means weight eligible frames equally; no predicted confidence filters the targets.','',
        '| Angle | Variant | Source | Heavy frames | XYZ mm | Depth mm | Output XYZ change mm |',
        '|---:|---|---|---:|---:|---:|---:|']
    for angle in (10,60):
        for name in rows[0]['variants']:
            for region in ('real','proxy'):
                values=[r['variants'][name]['regions'][region] for r in rows if r['angle']==angle and r['heavy'] and region in r['variants'][name]['regions']]
                v=dict(frames=len(values),**{k:statistics.mean(x[k] for x in values if x[k] is not None) for k in ('canonical_xyz_mm','depth_mm','xyz_change_mm','depth_change_mm')})
                result[f'{angle}/{name}/{region}']=v
                lines.append(f"| {angle} | {name} | {region} | {v['frames']} | {v['canonical_xyz_mm']:.4f} | {v['depth_mm']:.4f} | {v['xyz_change_mm']:.4f} |")
    stages=[]
    for stage in range(2):
        v=[r['variants']['normal']['hooks'][stage] for r in rows]
        stages.append(dict(patch_dtypes=sorted({x['patch_dtype'] for x in v}),**{k:statistics.mean(x[k] for x in v) for k in ('relative_write','changed_fraction','effective_relative_write')}))
    corrected=[r['correctable_points'] for r in rows]
    result['intervention_coverage']=dict(frames=64,zero_correctable_frames=sum(n==0 for n in corrected),minimum_points=min(corrected),mean_points=statistics.mean(corrected))
    result['write_stages']=stages
    lines+=['','Write/patch norm ratios: '+', '.join(f"{s['relative_write']:.6%}" for s in stages)+'.',
        'Patch addition dtypes: '+str([s['patch_dtypes'] for s in stages])+'.',
        'Oracle correspondence coverage: '+str(result['intervention_coverage'])+'.',
        '', 'Normal rewrite is bitwise identical to production output on every tested frame. Off/gain/oracle modes are frozen interventions; no training was performed.']
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(result['intervention_coverage']);print(stages)

if __name__=='__main__':main()
