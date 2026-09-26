"""Fixed natural, correct-reference pairs; visualization only."""
import json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from score_official_flow_v65 import sample_prediction
ROOT=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/official_flow_v65/experiment')
canvas=Image.new('RGB',(3*224,4*248),'white');draw=ImageDraw.Draw(canvas)
for rank in range(4):
 c=ROOT/'cache'/f'rank{rank}';x=torch.load(c/'input000.pt',weights_only=True)
 pred=torch.load(ROOT/'predictions'/f'rank{rank}/prediction000.pt',weights_only=True)['shared_rgb']
 label=torch.load(c/'labels000.pt',weights_only=True)
 q=x['query'][0];t=x['template'][0];mask=x['mask'][0]
 overlay=torch.where(mask[None],.5*q+.5*t,q)
 for column,(value,title) in enumerate(((q,'Real RGB'),(t,'GT-reference CAD'),(overlay,'Overlay + confident flow'))):
  im=Image.fromarray((value.permute(1,2,0).clamp(0,1).numpy()*255).round().astype('uint8'))
  canvas.paste(im,(column*224,rank*248+24));draw.text((column*224+3,rank*248+5),f'r{rank}: {title}',fill='black')
 uv,confidence,_=sample_prediction(pred,label['reference'])
 good=label['regions']['observed'][0]&(confidence[0]>=.3)
 for start,end in zip(label['reference'][0][good],uv[0][good]):
  x0,y0=start.tolist();x1,y1=end.tolist();origin=(448,rank*248+24)
  draw.line((x0+origin[0],y0+origin[1],x1+origin[0],y1+origin[1]),fill='magenta',width=1)
canvas.save(ROOT/'fixed_real_cad_pairs.png')
