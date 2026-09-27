"""Dense reference versus decoded and coarse-reference geometry on fixed targets."""
import argparse,json,statistics
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
rows=[]
for rank in range(8):
 f=a.root/f'rank{rank}'
 receipt=json.loads((f/'receipt.json').read_text());assert receipt['completed'] and receipt['records']==12
 part=[json.loads(s) for s in (f/'frames.jsonl').read_text().splitlines()];assert len(part)==12
 rows+=part
assert len({r['seed'] for r in rows})==96
for r in rows:
 for variant in r['variants'].values():
  for region,v in variant.items():
   base=r['variants']['network'][region]
   assert v['pixels']==base['pixels'] and v['covered_pixels']==base['covered_pixels']
lines=['# V73 dense input geometry audit','',
 'Frozen V72-fast checkpoint;96training-partition physical-holdout frames.32each at0/10/60degree controlled rotation error.',
 'All variants retain original target regions; unsupported reference pixels use the original network output.',
 'Patch mean uses validity-normalized14x14averaging. Patch sample uses the actual model anchor (including its patch-validity rule), repeated spatially.',
 'These explicit coarse-reference baselines are not a proof that learned256D tokens are equivalent to a mean coordinate.',
 'XYZ is canonical CAD error. Real-hidden depth retains original sensor targets; proxy depth uses rendered CAD targets. Observed depth here is a CAD-target diagnostic.', '',
 '| Angle | Heavy | Region | Variant | Frames | Reference coverage | XYZ mm | Depth mm |',
 '|---:|---|---|---|---:|---:|---:|---:|']
result={}
for angle in (0,10,60):
 for heavy in (False,True):
  for region in ('real','proxy','observed'):
   for name in rows[0]['variants']:
    values=[r['variants'][name][region] for r in rows if r['angle']==angle and r['heavy']==heavy and region in r['variants'][name]]
    if not values:continue
    item=dict(frames=len(values),coverage=statistics.mean(v['covered_pixels']/v['pixels'] for v in values))
    for key in ('canonical_xyz_mm','depth_mm','covered_xyz_mm','uncovered_xyz_mm','covered_depth_mm','uncovered_depth_mm'):
     vals=[v[key] for v in values if v[key] is not None];item[key]=statistics.mean(vals) if vals else None
    result[f'{angle}/{heavy}/{region}/{name}']=item
    lines.append(f"| {angle} | {heavy} | {region} | {name} | {len(values)} | {item['coverage']:.1%} | {item['canonical_xyz_mm']:.4f} | {item['depth_mm']:.4f} |")
result['invariants']=dict(frames=96,identical_target_and_reference_coverage_counts=True)
(a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n');(a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
