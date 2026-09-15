"""Matched real-initializer final1000 control versus dense rotation alignment."""
import argparse,json,sys
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash,save_init
from lip.engine.stream_config import make_model,load_stream_config


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--reference',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();torch.set_num_threads(2)
 e=json.loads((a.reference/'experiment.json').read_text());assert sha(e['parent'])==e['parent_sha256'];parent=torch.load(e['parent'],map_location='cpu',weights_only=False)
 assert e['parent_sha256']=='89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868'
 c=load_stream_config(e['arms']['real_mix']['config']);assert c['real_initialization_probability']==.5
 a.out.mkdir(parents=True,exist_ok=False);e=dict(e,arms={},source_sha256=source_hash(),protocol='Matched final1000 seed42: same retained residual weights, real PoseCNN train initializer mixture, fixed 64000 samples, S1O1, fresh optimizer, frozen RGB. Control and alignment both adapt non-RGB parent at identical learning rates.',
  intervention='Alignment additionally retains 14x14 fused observation tokens, independently encodes five CAD-only depth/silhouette/XYZ channels and cross-attends observation to CAD; zero-output rotation residual, unchanged center-head architecture. No rendered RGB texture in this first implementation.',
  evaluation='One online correction per frame in both trained arms. Full real-initialized val, compare startup rotation, center, all/severe accuracy. This isolates alignment from the separate frozen two-pass experiment.',
  promotion='Protect retained-parent overall ADD and strict ADD-S, severe strict, and first8 center. Require paired first8 rotation improvement versus both matched control and retained parent. Final1000 only. No new official test launch.')
 for name,enabled in [('control',False),('alignment',True)]:
  folder=a.out/name;folder.mkdir();cfg=dict(c,rotation_alignment=enabled,lr_rotation_alignment=1e-4,preflight_receipt=str((folder/'approval.json').resolve()))
  path=folder/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_stream_config(path)
  torch.manual_seed(42);model=make_model(cfg);state=model.state_dict()
  for key,value in parent['model'].items():assert key in state and value.shape==state[key].shape;state[key]=value.clone()
  model.load_state_dict(state,strict=True);assert all(torch.equal(t,model.state_dict()[k]) for k,t in parent['model'].items())
  if enabled:assert not model.rotation_alignment.output[-1].weight.count_nonzero() and not model.rotation_alignment.output[-1].bias.count_nonzero()
  statuses={k:(dict(status='initialized') if k.startswith('rotation_alignment.') else parent.get('migration_status',{}).get(k,dict(status='loaded'))) for k in state}
  migration=dict(parent=dict(path=e['parent'],sha256=e['parent_sha256'],architecture_id=parent['architecture_id'],new_stage_step=parent['new_stage_step'],split_hash=e['split_hash'],mesh_hash=e['mesh_hash']),parameters=statuses,all_parent_tensors_bitwise_equal=True,optimizer_restored=False)
  model.migration_status=statuses;save_init(folder/'init.pt',model,cfg,migration)
  e['arms'][name]=dict(config=str(path.resolve()),init=str((folder/'init.pt').resolve()),init_sha256=sha(folder/'init.pt'))
 (a.out/'experiment.json').write_text(json.dumps(e,indent=2));print(e['source_sha256'])

if __name__=='__main__':main()
