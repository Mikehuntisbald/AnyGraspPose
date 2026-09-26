"""Score cached predictions after inference; no confidence filtering of targets."""
import argparse,json,statistics
from pathlib import Path
import torch
from torch.nn import functional as F

def sample_prediction(pred,reference):
    # Same pixel-center resize convention as bilinear224->280 input resize.
    uv=(reference+.5)*1.25-.5
    grid=2*(uv+.5)/280-1
    sample=lambda x:F.grid_sample(x.float(),grid[:,None],align_corners=False)[:,:,0].transpose(1,2)
    mass=sample(pred['mask'][:,None])
    flow=sample(pred['flow'])/mass.clamp_min(1e-8)
    confidence=sample(pred['confidence'][:,None])/mass.clamp_min(1e-8)
    # Zero coverage remains in the denominator and produces zero-flow fallback.
    return reference+flow/1.25,confidence[...,0],mass[...,0]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 rows=[]
 for rank in range(8):
  cache=a.root/'cache'/f'rank{rank}';pred=a.root/'predictions'/f'rank{rank}'
  assert json.loads((pred/'receipt.json').read_text())['completed']
  records=[json.loads(s) for s in (cache/'frames.jsonl').read_text().splitlines()]
  for i,row in enumerate(records):
   labels=torch.load(cache/f'labels{i:03d}.pt',weights_only=True)
   prediction=torch.load(pred/f'prediction{i:03d}.pt',weights_only=True)
   methods={'zero':labels['reference'],'jepa_v60':labels['baseline']};coverage={}
   for name,value in prediction.items():
    methods[name],_,coverage[name]=sample_prediction(value,labels['reference'])
   item=dict(row,metrics={})
   for method,uv in methods.items():
    error=(uv-labels['truth']).norm(dim=-1)
    item['metrics'][method]={}
    for region,mask in labels['regions'].items():
     if not mask.any():continue
     values=error[mask]
     item['metrics'][method][region]=dict(points=int(mask.sum()),epe=float(values.mean()),within3=float((values<=3).float().mean()),
       zero_coverage=int((coverage[method][mask]<=1e-8).sum()) if method in coverage else 0)
   rows.append(item)
 assert len(rows)==96
 (a.root/'scored_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 summary={};lines=['# V65 official pretrained flow versus JEPA local flow','',
 '96 physical-holdout training-partition frames;32 each at0/10/60 degrees. Frozen, same student RGB/crop/reference geometry.',
 'Official model is RGB-only, with different pretraining/architecture; this is not an equal-training-budget ablation or official benchmark reproduction.',
 '280px predictions sampled back at224px source points using pixel-center mapping and valid-mask-normalized interpolation.',
 'No confidence threshold filters any evaluation point. Labels are separate from inputs and read only by this scorer.','',
 '| Case | Region | Method | Frames | EPE px | Within3px |',
 '|---|---|---|---:|---:|---:|']
 for angle in (0,10,60):
  for heavy in (False,True):
   for region in ('observed','real','proxy'):
    for method in ('zero','jepa_v60','shared_rgb','official_gray'):
     values=[r['metrics'][method][region] for r in rows if r['base_kind']==f'controlled_{angle}' and r['heavy']==heavy and region in r['metrics'][method]]
     if not values:continue
     key=f'{angle}/{heavy}/{region}/{method}'
     v=dict(frames=len(values),points=sum(x['points'] for x in values),epe=statistics.mean(x['epe'] for x in values),within3=statistics.mean(x['within3'] for x in values),zero_coverage=sum(x['zero_coverage'] for x in values))
     summary[key]=v;lines.append(f"| {angle}deg {'heavy' if heavy else 'nonheavy'} | {region} | {method} | {v['frames']} | {v['epe']:.3f} | {v['within3']:.1%} |")
 (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n');(a.root/'outcome.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
