"""Separate canonical-prior quality, coverage and final atlas geometry."""
import argparse,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=[]
    for rank in range(8):
        folder=a.root/f'rank{rank}'
        receipt=json.loads((folder/'receipt.json').read_text())
        assert receipt['completed'] and receipt['normal_forward_bitwise_verified']
        shard=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
        assert len(shard)==receipt['records']==8
        rows+=shard
    assert len({r['seed'] for r in rows})==64
    for r in rows:
        normal=r['variants']['normal']
        for name,v in r['variants'].items():
            assert v['flow']==normal['flow']
            for region,x in v['regions'].items():
                base=normal['regions'][region]
                assert x['pixels']==base['pixels'] and x.get('canonical_pixels')==base.get('canonical_pixels')
                assert x['depth_change_mm']==0 and x['depth_mm']==base['depth_mm']
    lines=['# Canonical prior quality and atlas selection diagnostic', '',
           'Frozen V60-extra;64physical-holdout frames inside the training partition. GT prior/endpoints are diagnostic only.',
           'Primary errors include all original target pixels. Covered-only errors are secondary diagnostics, not replacement metrics.',
           'Coverage is mean per-frame target coverage; prior/final errors are means over eligible frames. No pose/depth improvement claim.', '',
           '| Angle | Heavy | Region | Variant | Frames | Coverage | Prior XYZ mm | Final XYZ mm | Covered prior mm | Covered final mm |',
           '|---:|---|---|---|---:|---:|---:|---:|---:|---:|']
    result={}
    for angle in (10,60):
        for heavy in (False,True):
            for region in ('observed','real','proxy'):
                for name in rows[0]['variants']:
                    values=[r['variants'][name]['regions'][region] for r in rows if r['angle']==angle and r['heavy']==heavy
                            and region in r['variants'][name]['regions'] and r['variants'][name]['regions'][region].get('canonical_pixels',0)>0]
                    if not values:continue
                    item=dict(frames=len(values),coverage=statistics.mean(v['covered_pixels']/v['canonical_pixels'] for v in values))
                    for k in ('prior_xyz_mm','canonical_xyz_mm','covered_prior_xyz_mm','covered_final_xyz_mm','uncovered_prior_xyz_mm','uncovered_final_xyz_mm'):
                        available=[v[k] for v in values if v[k] is not None]
                        item[k]=statistics.mean(available) if available else None
                        item[k+'_frames']=len(available)
                    result[f'{angle}/{heavy}/{region}/{name}']=item
                    fmt=lambda v:'n/a' if v is None else f'{v:.3f}'
                    numbers=' | '.join(fmt(item[k]) for k in ('prior_xyz_mm','canonical_xyz_mm','covered_prior_xyz_mm','covered_final_xyz_mm'))
                    lines.append(f"| {angle} | {heavy} | {region} | {name} | {len(values)} | {item['coverage']:.1%} | {numbers} |")
    result['invariants']=dict(frames=64,depth_unchanged=True,flow_unchanged=True,target_counts_unchanged=True)
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
