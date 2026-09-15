"""Two matched stages: continued R1K1 vs added dense spatial keyframe memory."""
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import make_model,load_stream_config
from lip.engine.stream_checkpoint import sha,source_hash,save_init
from lip.engine.config import check_data_gate
from lip.data.stream_clips import StreamClips


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('parent','data-root','index-root','initial-poses','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--temporal-occlusion-probability',type=float,default=0.)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);audit=check_data_gate(a.index_root)
    expected='10f202a288a58a84b22e0e07c45278ec370af1202ff9acc4544e0288a2773c0a'
    assert sha(a.parent)==expected
    parent=torch.load(a.parent,map_location='cpu',weights_only=False);assert parent['architecture_id']=='stream_rk_factorial' and parent['new_stage_step']==1000
    c=parent['config'].copy();c.update(world_size=4,batch_sequences_per_gpu=16,grad_accum_steps=1,effective_sequences_per_step=64,
        supervised_unroll_frames=48,nominal_supervised_updates_per_step=3072,max_stage_steps=1000,save_every=250,warmup_steps=100)
    if a.temporal_occlusion_probability:c['temporal_occlusion_probability']=a.temporal_occlusion_probability
    e=dict(parent=str(a.parent.resolve()),parent_sha256=expected,source_sha256=source_hash(),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],
        data_root=str(a.data_root.resolve()),index_root=str(a.index_root.resolve()),initial_poses=str(a.initial_poses.resolve()),initial_poses_sha256=sha(a.initial_poses),
        seed=42,steps=1000,arms={},protocol='Same 64 fragments/step, 8 burn-in + 48 supervised frames, same old-parameter learning rates; final step 1000 fixed in advance',
        trained_modules='Existing R1K1 trainable modules in both arms; added patch attention/gate/pose-conditioning only in spatial',
        quality_head_enabled=False,foundationpose_calls=0,test_access=False,reference_evaluation=str(a.parent.parent.parent/'s0_val'))
    e['temporal_occlusion_probability']=a.temporal_occlusion_probability
    e['occlusion_scope']='Training only: causal initial-observation plan, depth-tested RGB-D compositing, no current/future GT in augmentation; validation observations unchanged'
    for name in ('control','spatial'):
        folder=a.out/name;folder.mkdir();cfg=dict(c,preflight_receipt=str((folder/'approval.json').resolve()))
        if name=='spatial':cfg.update(architecture_id='stream_rk_spatial',spatial_memory_side=14)
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_stream_config(path)
        torch.manual_seed(42);model=make_model(cfg);loaded=model.load_state_dict(parent['model'],strict=False)
        assert not loaded.unexpected_keys and all(k.startswith(('patch_attention.','patch_gate.','patch_pose.')) for k in loaded.missing_keys)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in parent['model'].items())
        migration=dict(parent=dict(path=str(a.parent.resolve()),sha256=expected,architecture_id=parent['architecture_id'],new_stage_step=1000,
            split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash']),parameters={k:dict(status='loaded' if k in parent['model'] else 'initialized') for k in model.state_dict()},
            optimizer_restored=False,new_stage_step=0,all_parent_tensors_bitwise_equal=True)
        model.migration_status=migration['parameters'];save_init(folder/'init.pt',model,cfg,migration)
        e['arms'][name]=dict(config=str(path.resolve()),init=str((folder/'init.pt').resolve()),init_sha256=sha(folder/'init.pt'),trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad})
    ds=StreamClips(a.data_root,a.index_root,8,48,seed=42)
    assert set(ds.groups)=={s['object_id'] for s in ds.streams},'Longer fragments must retain the entire training object scope'
    e['eligible_training_objects']=sorted(ds.groups);e['eligible_streams']=sum(bool(len(v)) for v in ds.starts);e['total_training_streams']=len(ds.streams)
    manifest=a.out/'training_samples.json';manifest.write_text(json.dumps([ds.choose(i) for i in range(64000)]))
    e['training_manifest']=str(manifest.resolve());e['training_manifest_sha256']=sha(manifest)
    (a.out/'experiment.json').write_text(json.dumps(e,indent=2));print(json.dumps({k:v for k,v in e.items() if k!='arms'},indent=2))


if __name__=='__main__':main()
