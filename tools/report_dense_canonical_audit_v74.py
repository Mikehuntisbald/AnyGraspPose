import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
assert json.loads((a.root/'status.json').read_text())['primary_records_exact']
lines=['# V74 dense backward-correspondence audit','',
 '128repeated cases; primary geometry and existing sparse-flow records match the original evaluation exactly.',
 'Dense flow maps observation pixels to the estimated CAD raster; it is not the sparse forward flow in the main report.',
 'EPE is measured on fixed GT-supported correspondences, without predicted-gate filtering. Gate fraction uses all region pixels.', '',
 '| Angle | Heavy | Stage | Region | Frames | GT support | Gate on | Dense EPE px | Zero-flow EPE px |',
 '|---:|---|---:|---|---:|---:|---:|---:|---:|']
out={}
for angle in (0,10,60):
 rows=[r for f in (a.root/f'angle{angle}').glob('rank*/frames.jsonl') for r in map(json.loads,f.read_text().splitlines())]
 for heavy in (False,True):
  for stage in (0,1):
   for region in ('real','proxy'):
    key=f'{stage}/{region}'
    vals=[r['dense_canonical'][key] for r in rows if r['heavy']==heavy and key in r['dense_canonical']]
    if not vals:continue
    item=dict(frames=len(vals),support=statistics.mean(v['supported_pixels']/v['pixels'] for v in vals),gate=statistics.mean(v['gate_fraction'] for v in vals))
    for k in ('epe','zero_epe','warped_xyz_mm'):
     available=[v[k] for v in vals if v[k] is not None];item[k]=statistics.mean(available) if available else None
    out[f'{angle}/{heavy}/{stage}/{region}']=item
    if item['epe'] is not None:lines.append(f"| {angle} | {heavy} | {stage} | {region} | {len(vals)} | {item['support']:.1%} | {item['gate']:.1%} | {item['epe']:.4f} | {item['zero_epe']:.4f} |")
(a.root/'outcome.json').write_text(json.dumps(out,indent=2)+'\n');(a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
