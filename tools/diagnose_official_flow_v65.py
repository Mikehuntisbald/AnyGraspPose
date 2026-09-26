"""Published0.3 confidence threshold: conditional diagnostics, never target filtering."""
import argparse,json
from pathlib import Path
import torch
from score_official_flow_v65 import sample_prediction

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 groups={};identical=0
 for rank in range(8):
  cache=a.root/'cache'/f'rank{rank}';pred=a.root/'predictions'/f'rank{rank}'
  records=[json.loads(s) for s in (cache/'frames.jsonl').read_text().splitlines()]
  for i,row in enumerate(records):
   label=torch.load(cache/f'labels{i:03d}.pt',weights_only=True);result=torch.load(pred/f'prediction{i:03d}.pt',weights_only=True)
   identical+=int(all(torch.equal(result['shared_rgb'][k],result['official_gray'][k]) for k in ('flow','confidence','mask')))
   uv,confidence,_=sample_prediction(result['shared_rgb'],label['reference'])
   error=(uv-label['truth']).norm(dim=-1);known=torch.stack(list(label['regions'].values())).any(0)
   selected=known&(confidence>=.3);visible=label['regions']['observed'];good=visible&(error<=3)
   key=f"{row['base_kind']}/{'heavy' if row['heavy'] else 'nonheavy'}"
   g=groups.setdefault(key,dict(frames=0,frames_with_selected=0,known=0,selected=0,visible=0,visible_selected=0,good_selected=0,error_sum=0.))
   g['frames']+=1;g['frames_with_selected']+=int(selected.any());g['known']+=int(known.sum());g['selected']+=int(selected.sum());g['visible']+=int(visible.sum())
   g['visible_selected']+=int((selected&visible).sum());g['good_selected']+=int((selected&good).sum());g['error_sum']+=float(error[selected].sum())
 for g in groups.values():
  g['coverage']=g['selected']/max(g['known'],1);g['selected_epe']=g['error_sum']/g['selected'] if g['selected'] else None
  g['selected_visibility_precision']=g['visible_selected']/g['selected'] if g['selected'] else None
  g['selected_visible_and_3px_precision']=g['good_selected']/g['selected'] if g['selected'] else None
  g['visible_recall']=g['visible_selected']/g['visible'] if g['visible'] else None
 output=dict(threshold=.3,threshold_source='Official GoTrackOpts.visib_threshold',identical_background_variants=identical,groups=groups,
  interpretation='Conditional selected-point diagnostics only. Primary unfiltered errors remain in REPORT.md. Not recovered geometry or pose accuracy.')
 (a.root/'confidence_diagnostic.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
