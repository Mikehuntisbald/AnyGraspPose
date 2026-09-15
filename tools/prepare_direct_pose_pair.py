"""Prepare a matched parent continuation versus zero-start direct pose heads."""
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.config import check_data_gate
from lip.engine.stream_checkpoint import sha,source_hash,save_init
from lip.engine.stream_config import make_model,load_stream_config
from lip.data.stream_sampling import load_fixed_manifest


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('parent-experiment','manifest','data-root','index-root','out'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--parent-arm',required=True);p.add_argument('--expected-parent-sha',required=True)
    p.add_argument('--expected-manifest-sha',required=True);a=p.parse_args()
    old=json.loads((a.parent_experiment/'experiment.json').read_text());assert json.loads((a.parent_experiment/'status.json').read_text())['phase']=='completed'
    parent_path=a.parent_experiment/a.parent_arm/'train/last.pt';assert sha(parent_path)==a.expected_parent_sha
    parent=torch.load(parent_path,map_location='cpu',weights_only=False);assert parent['architecture_id']=='stream_rk_spatial'
    reference=a.parent_experiment/a.parent_arm/'s0_val';evaluation=json.loads((reference/'manifest.json').read_text())
    assert evaluation['completed'] and evaluation['population_verified'] and evaluation['checkpoint_sha256']==a.expected_parent_sha
    assert evaluation['frames']==23200 and len(evaluation['streams'])==320 and evaluation['fp_calls']==0
    samples,sampling=load_fixed_manifest(a.manifest,a.expected_manifest_sha);assert len(samples)==64000
    audit=check_data_gate(a.index_root);assert all(evaluation[k]==audit[k] for k in ('split_hash','mesh_hash'))
    a.out.mkdir(parents=True,exist_ok=False);manifest=a.out/'training_samples.json';manifest.write_bytes(a.manifest.read_bytes())
    c=dict(parent['config'],world_size=4,batch_sequences_per_gpu=16,grad_accum_steps=1,effective_sequences_per_step=64,
        supervised_unroll_frames=48,nominal_supervised_updates_per_step=3072,max_stage_steps=1000,save_every=250,warmup_steps=100,
        fixed_sampling_manifest_sha256=a.expected_manifest_sha)
    assert c['temporal_occlusion_probability']==.5
    e=dict(parent=str(parent_path.resolve()),parent_sha256=a.expected_parent_sha,source_sha256=source_hash(),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],
        data_root=str(a.data_root.resolve()),index_root=str(a.index_root.resolve()),initial_poses=old['initial_poses'],initial_poses_sha256=old['initial_poses_sha256'],
        seed=42,steps=1000,reader_arm='direct_pose',arms={},training_manifest=str(manifest.resolve()),training_manifest_sha256=a.expected_manifest_sha,
        temporal_occlusion_probability=.5,reference_evaluation=str(reference.resolve()),quality_head_enabled=False,foundationpose_calls=0,test_access=False,
        protocol='Identical sampled RGB-D fragments, noise seeds and augmentation schedule, same trained spatial parent, 64 fragments/step, 8+48 frames, fixed final step 1000.',
        trained_modules='Existing small modules in both arms; direct arm adds separate zero-initialized rotation and center action residual heads. Parent backbone, temporal encoder and original pose head remain frozen.',
        action_semantics='Residual rotation-vector radians and center displacement/d are added to parent actions before the existing Update(T_base,delta). No SE(3) exponential or camera-origin rotation of center translation.')
    for name in ('control','direct_pose'):
        folder=a.out/name;folder.mkdir();cfg=dict(c,preflight_receipt=str((folder/'approval.json').resolve()))
        if name=='direct_pose':cfg['architecture_id']='stream_rk_direct_pose'
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_stream_config(path)
        torch.manual_seed(42);model=make_model(cfg);loaded=model.load_state_dict(parent['model'],strict=False)
        assert not loaded.unexpected_keys and all(k.startswith('direct_pose_residual.') for k in loaded.missing_keys)
        assert all(torch.equal(t,model.state_dict()[key]) for key,t in parent['model'].items())
        migration=dict(parent=dict(path=str(parent_path.resolve()),sha256=a.expected_parent_sha,architecture_id=parent['architecture_id'],new_stage_step=parent['new_stage_step'],
            split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash']),parameters={k:dict(status='loaded' if k in parent['model'] else 'initialized') for k in model.state_dict()},
            optimizer_restored=False,new_stage_step=0,all_parent_tensors_bitwise_equal=True)
        model.migration_status=migration['parameters'];save_init(folder/'init.pt',model,cfg,migration)
        e['arms'][name]=dict(config=str(path.resolve()),init=str((folder/'init.pt').resolve()),init_sha256=sha(folder/'init.pt'),
            trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad})
    old_names=set(e['arms']['control']['trainable']);new_names=set(e['arms']['direct_pose']['trainable'])
    assert old_names<=new_names and all(n.startswith('direct_pose_residual.') for n in new_names-old_names)
    (a.out/'experiment.json').write_text(json.dumps(e,indent=2));print(json.dumps({k:v for k,v in e.items() if k!='arms'},indent=2))


if __name__=='__main__':main()
