"""Paired final-only routing versus historical matched V62 strong control."""
import argparse,json,statistics
from pathlib import Path

def load(folder):
 rows=[]
 for rank in range(8):
  p=folder/f'rank{rank}'
  receipt=json.loads((p/'receipt.json').read_text());assert receipt['completed']
  rows.extend(json.loads(s) for s in (p/'frames.jsonl').read_text().splitlines())
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control-root',type=Path,required=True);a=p.parse_args()
 assert json.loads((a.root/'control_replay_exact.json').read_text())['loss_grad_metrics_data_exact']
 lines=['# V64 final-latent-only DPT routing','',
 '200 candidate updates from the same V60 source as V62 strong200. Same data, gain100, objective and schedule.',
 'Historical control replay reproduced first2 updates exactly across8 ranks; full200-step control is reused, not rerun.',
 'Physical training holdout;32/64/32 controlled observations at0/10/60 degrees; no confidence filtering of target regions.','',
 '| Angle | Model | Source | Heavy frames | Canonical XYZ mm | Depth mm | Flow EPE px (all eligible frames) |',
 '|---:|---|---|---:|---:|---:|---:|']
 result={}
 for angle in (0,10,60):
  data={'baseline':load(a.control_root/'probe'/f'baseline_{angle}'),
        'matched_control':load(a.control_root/'probe'/f'strong_{angle}'),
        'final_only':load(a.root/'probe'/f'final_{angle}')}
  identity=[(r['seed'],r['stream'],r['heavy'],r['natural']) for r in data['baseline']]
  for name,rows in data.items():
   assert [(r['seed'],r['stream'],r['heavy'],r['natural']) for r in rows]==identity
   for region in ('real','proxy'):
    values=[r['metrics'][region] for r in rows if r['heavy'] and r['metrics'][region]]
    flow=[r['flow'][f'flow1_{region}_epe'] for r in rows if r['flow'][f'flow_{region}_count']>0]
    v=dict(frames=len(values),canonical_xyz_mm=statistics.mean(x['canonical_xyz_mm'] for x in values),
      depth_mm=statistics.mean(x['depth_mm'] for x in values),flow_epe=statistics.mean(flow))
    result[f'{angle}/{name}/{region}']=v
    lines.append(f"| {angle} | {name} | {region} | {v['frames']} | {v['canonical_xyz_mm']:.3f} | {v['depth_mm']:.3f} | {v['flow_epe']:.3f} |")
 (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n');(a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
 print('\n'.join(lines))

if __name__=='__main__':main()
