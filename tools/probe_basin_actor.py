"""Isolated real-data gradient probe; does not update or resume the production actor."""
import json,os,torch,yaml
from pathlib import Path
from lip.models.tracker import Tracker
from lip.models.basin import load_frozen_basin,basin_loss
from lip.data.clips import ClipDataset
from lip.geometry.renderer import Renderer
from lip.engine.runtime import batch_step
from lip.integrations.frozen_fp import FrozenFoundationPose
from lip.losses import pose_loss
r=Path('/mnt/why/dexycb_lip');os.chdir(r);j=r/'runs/basin_dual_v2';torch.set_num_threads(2);torch.manual_seed(731)
c=yaml.safe_load((r/'runs/fpaware_19000/candidate/configs/fpaware_19000.yaml').read_text());actor=Tracker(False).cuda();ck=torch.load(r/'runs/lip_v1_s0/last.pt',map_location='cpu',weights_only=False);actor.load_state_dict(ck['model']);actor.train()
q=load_frozen_basin(j/'fit/critic.pt','cuda',allow_experimental=True);root=os.environ['DEX_YCB_DIR'];fp=FrozenFoundationPose(c['foundationpose_root'],root,torch.device('cuda',0),c['foundationpose_refiner_sha256'])
ds=ClipDataset(root,'cache/dexycb_s0',length=c['clip_length'],steps=32,augmentation=True);items=[ds[i] for i in range(32)];diagnostic={}
def observe(u,inp,out):
 if u:return
 targets=torch.stack([x['poses'][c['clip_length']].cuda() for x in items]);pts=torch.stack([torch.as_tensor(x['mesh']['points'],device='cuda') for x in items])
 lp,_=pose_loss(out['pose_centered'],targets,pts,inp['object_diameter_m']);lb=basin_loss(q,out['latent'],torch.cat((out['delta_rotvec'],out['delta_center_norm']),-1));parameter=actor.head[-1].weight
 gp=torch.autograd.grad(lp,parameter,retain_graph=True)[0];gb=torch.autograd.grad(lb,parameter,retain_graph=True)[0]
 diagnostic.update(pose_head_gradient_norm=float(gp.norm()),basin_head_gradient_norm=float(gb.norm()),weighted_gradient_ratio_at_0_001=float(.001*gb.norm()/gp.norm().clamp_min(1e-12)),gradient_cosine=float(torch.nn.functional.cosine_similarity(gp.flatten(),gb.flatten(),dim=0)))
values,_=batch_step(actor,items,Renderer('cuda'),c,4,True,True,history_mode='lip_fp',fp_transition=fp,basin_critic=q,basin_weight=.001,observer=observe)
assert all(p.grad is None for p in q.parameters());assert all(p.grad is None for p in fp.refiner.model.parameters());assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
receipt=dict(passed=True,scope='engineering-only probe; critic quality gate remains failed; no production update',actor_step=ck['global_step'],batch=32,rollout=4,weight=.001,peak_bytes=torch.cuda.max_memory_allocated(),critic_frozen=True,fp_frozen=True,values=values,gradient_diagnostic=diagnostic)
(j/'gradient_probe.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
