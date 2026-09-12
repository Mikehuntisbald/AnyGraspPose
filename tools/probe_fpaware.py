import argparse,json,os,torch,yaml,time
from pathlib import Path
from lip.models.tracker import Tracker
from lip.data.clips import ClipDataset
from lip.engine.runtime import batch_step
from lip.geometry.renderer import Renderer
from lip.integrations.frozen_fp import FrozenFoundationPose
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--batch',type=int,default=32);a=p.parse_args()
c=yaml.safe_load(Path(a.config).read_text());torch.set_num_threads(2);root=os.environ['DEX_YCB_DIR']
model=Tracker(False).cuda();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.train()
env=FrozenFoundationPose(c['foundationpose_root'],root,torch.device('cuda',0));ds=ClipDataset(root,'cache/dexycb_s0',length=c['clip_length'],steps=a.batch,augmentation=True)
items=[ds[i] for i in range(a.batch)];renderer=Renderer('cuda');rows=[]
for mode in ['noisy_gt','lip_only','lip_fp']:
 model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();begin=time.time()
 values,_=batch_step(model,items,renderer,c,4,True,True,history_mode=mode,fp_transition=env)
 assert all(p.grad is None for p in env.refiner.model.parameters())
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
 row=dict(mode=mode,batch=a.batch,rollout=4,seconds=time.time()-begin,peak_bytes=torch.cuda.max_memory_allocated(),fp_frozen=True,lip_gradients_finite=True,**values);rows.append(row);print(json.dumps(row),flush=True)
Path(a.out).write_text(json.dumps(dict(passed=True,checkpoint_step=ck['global_step'],rows=rows),indent=2))
