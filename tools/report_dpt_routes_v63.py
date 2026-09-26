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
 summary={}
 for name in ('depth','xyz_surrogate'):
  summary[name]=[statistics.mean(r['variants']['gradients']['sensitivity'][name]['fraction'][i] for r in rows) for i in range(4)]
 lines=['# V63 DPT route sensitivity','',
 '64 frozen training-partition physical-holdout frames;10/60 degree controlled references. All targets retained.',
 'Gradient shares are input-norm-scaled local sensitivities, not information fractions. XYZ uses the existing straight-through local surrogate.',
 'Roll means a14px horizontal shift of one input lattice; this is a frozen intervention, not a proposed trained model.','',
 '| Input / DPT scale | Depth gradient share | XYZ surrogate share | Heavy10 real frames | XYZ output change mm | Depth output change mm |',
 '|---|---:|---:|---:|---:|---:|']
 for i,side in enumerate((64,32,16,8)):
  values=[r['variants'][f'roll{i}']['regions']['real'] for r in rows if r['heavy'] and r['angle']==10 and 'real' in r['variants'][f'roll{i}']['regions']]
  xyz=statistics.mean(v['xyz_change_mm'] for v in values);depth=statistics.mean(v['depth_change_mm'] for v in values)
  summary[f'roll{i}']=dict(frames=len(values),xyz_change_mm=xyz,depth_change_mm=depth)
  lines.append(f"| {i} / {side}x{side} | {summary['depth'][i]:.2%} | {summary['xyz_surrogate'][i]:.2%} | {len(values)} | {xyz:.3f} | {depth:.3f} |")
 lines+=['','Normal and gradient-capture forward values are bitwise equal to the baseline for all64 frames.',
 'Earlier branches dominate local depth sensitivity. The last branch still affects geometry; the result does not prove it is unused.']
 (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n');(a.root/'outcome.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
