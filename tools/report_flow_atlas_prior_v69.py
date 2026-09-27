"""Report frozen decoder-prior interventions without changing evaluation masks."""
import argparse,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=[]
    for rank in range(8):
        folder=a.root/f'rank{rank}'
        receipt=json.loads((folder/'receipt.json').read_text())
        assert receipt['completed'] and receipt['normal_forward_bitwise_verified']
        part=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
        assert len(part)==receipt['records']==8
        rows+=part
    assert len({r['seed'] for r in rows})==64
    for r in rows:
        normal=r['variants']['normal']
        for name,v in r['variants'].items():
            assert v['flow']==normal['flow']
            for region,value in v['regions'].items():
                assert value['pixels']==normal['regions'][region]['pixels']
                assert value['depth_change_mm']==0
                assert value['depth_mm']==normal['regions'][region]['depth_mm']
    lines=['# V69 frozen flow-aligned canonical prior', '',
           '64 physical-holdout training-partition frames, frozen V60-extra. Same seeds as V66/V67.',
           'Oracle endpoints and oracle-supported anchor selection are diagnostic only. No training or pose solver.',
           'Flow/depth are bitwise unchanged. All original target pixels remain in evaluation; no coverage filtering.', '',
           '| Angle | Heavy | Region | Variant | Frames | Canonical XYZ mm | Depth mm |',
           '|---:|---|---|---|---:|---:|---:|']
    result={}
    for angle in (10,60):
        for heavy in (False,True):
            for region in ('real','proxy'):
                for name in rows[0]['variants']:
                    values=[r['variants'][name]['regions'][region] for r in rows
                            if r['angle']==angle and r['heavy']==heavy and region in r['variants'][name]['regions']]
                    if not values:continue
                    xyz=[v['canonical_xyz_mm'] for v in values if v['canonical_xyz_mm'] is not None]
                    item=dict(frames=len(values),canonical_frames=len(xyz),canonical_xyz_mm=statistics.mean(xyz) if xyz else None,
                              depth_mm=statistics.mean(v['depth_mm'] for v in values))
                    result[f'{angle}/{heavy}/{region}/{name}']=item
                    if xyz:lines.append(f"| {angle} | {heavy} | {region} | {name} | {len(values)} | {item['canonical_xyz_mm']:.4f} | {item['depth_mm']:.4f} |")
    result['invariants']=dict(frames=64,depth_unchanged=True,flow_unchanged=True,target_counts_unchanged=True)
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
