"""Matched continued training, changing only the offline fragment sampler."""
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lip.engine.config import check_data_gate
from lip.engine.stream_checkpoint import source_hash,sha,save_init
from lip.engine.stream_config import load_stream_config,make_model


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('parent-experiment','manifests','data-root','index-root','out'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--parent-arm',required=True);p.add_argument('--expected-parent-sha',required=True)
    a=p.parse_args();old=json.loads((a.parent_experiment/'experiment.json').read_text())
    assert json.loads((a.parent_experiment/'status.json').read_text())['phase']=='completed'
    parent_path=a.parent_experiment/a.parent_arm/'train/last.pt';assert sha(parent_path)==a.expected_parent_sha
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    assert parent['architecture_id'] in ('stream_rk_spatial','stream_rk_aligned') and parent['new_stage_step']==1000
    audit=check_data_gate(a.index_root);sampling=json.loads((a.manifests/'receipt.json').read_text())
    assert sampling['completed'] and sampling['uniform_matches_all_dataset_draws'] and sampling['object_and_noise_schedules_identical']
    assert sampling['split_hash']==audit['split_hash'] and sampling['mesh_hash']==audit['mesh_hash']
    assert sampling['max_hard_branch_repeats_per_window']==8 and sampling['sample_counter_range']==[128000,191999]
    a.out.mkdir(parents=True,exist_ok=False)
    c=dict(parent['config'],world_size=4,batch_sequences_per_gpu=16,grad_accum_steps=1,effective_sequences_per_step=64,
        supervised_unroll_frames=48,nominal_supervised_updates_per_step=3072,max_stage_steps=1000,save_every=250,warmup_steps=100)
    assert c['temporal_occlusion_probability']==.5
    e=dict(parent=str(parent_path.resolve()),parent_sha256=a.expected_parent_sha,source_sha256=source_hash(),
        split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],data_root=str(a.data_root.resolve()),index_root=str(a.index_root.resolve()),
        initial_poses=old['initial_poses'],initial_poses_sha256=old['initial_poses_sha256'],seed=42,steps=1000,arms={},reader_arm='natural',
        temporal_occlusion_probability=.5,reference_evaluation=str((a.parent_experiment/a.parent_arm/'s0_val').resolve()),
        sampling_receipt=sampling,sampling_selections=str((a.manifests/'selections.jsonl').resolve()),
        protocol='Same frozen parent function and trainable modules; only offline train-fragment sampling differs. Same per-position object/noise seed schedule, 64 fragments/step, 8+48 frames, final step 1000, no intermediate val selection.',
        trained_modules='Existing small modules only; original RGB/geometry/spatial fusion/temporal/pose-head core remains frozen in both arms.',
        quality_head_enabled=False,foundationpose_calls=0,test_access=False,optimizer_restored=False)
    for name in ('control','natural'):
        manifest=a.manifests/(name+'.json');assert sha(manifest)==sampling['arms'][name]['manifest_sha256']
        folder=a.out/name;folder.mkdir();cfg=dict(c,preflight_receipt=str((folder/'approval.json').resolve()),fixed_sampling_manifest_sha256=sha(manifest))
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_stream_config(path)
        torch.manual_seed(42);model=make_model(cfg);model.load_state_dict(parent['model'],strict=True)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in parent['model'].items())
        migration=dict(parent=dict(path=str(parent_path.resolve()),sha256=a.expected_parent_sha,architecture_id=parent['architecture_id'],
            new_stage_step=1000,split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash']),
            parameters={k:dict(status='loaded') for k in model.state_dict()},optimizer_restored=False,new_stage_step=0,all_parent_tensors_bitwise_equal=True)
        model.migration_status=migration['parameters'];save_init(folder/'init.pt',model,cfg,migration)
        dest=folder/'training_samples.json';dest.write_bytes(manifest.read_bytes())
        e['arms'][name]=dict(config=str(path.resolve()),init=str((folder/'init.pt').resolve()),init_sha256=sha(folder/'init.pt'),
            training_manifest=str(dest.resolve()),training_manifest_sha256=sha(dest),trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad})
    assert e['arms']['control']['trainable']==e['arms']['natural']['trainable']
    e['training_manifest']=e['arms']['control']['training_manifest'];e['training_manifest_sha256']=e['arms']['control']['training_manifest_sha256']
    (a.out/'experiment.json').write_text(json.dumps(e,indent=2));print(json.dumps({k:v for k,v in e.items() if k not in ('arms','sampling_receipt')},indent=2))


if __name__=='__main__':main()
