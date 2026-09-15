"""Prepare startup supervision x startup occlusion on one identical zero-start actor."""
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
from lip.engine.stream_training import supervision_positions
from lip.data.stream_sampling import load_fixed_manifest

ARMS=('S0O0','S0O1','S1O0','S1O1')


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('parent-experiment','manifest','data-root','index-root','out'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--expected-parent-sha',required=True);p.add_argument('--expected-manifest-sha',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--manifest-receipt',type=Path)
    a=p.parse_args()
    if not 0<=a.seed<2**31:p.error('Seed must be in [0, 2**31)')
    if a.seed!=42 and a.manifest_receipt is None:p.error('An independent seed requires its sampling receipt')
    old=json.loads((a.parent_experiment/'experiment.json').read_text())
    assert json.loads((a.parent_experiment/'status.json').read_text())['phase']=='completed'
    parent_path=a.parent_experiment/'spatial/train/last.pt';assert sha(parent_path)==a.expected_parent_sha
    parent=torch.load(parent_path,map_location='cpu',weights_only=False);assert parent['architecture_id']=='stream_rk_spatial'
    reference=a.parent_experiment/'spatial/s0_val';ev=json.loads((reference/'manifest.json').read_text())
    assert ev['completed'] and ev['population_verified'] and ev['checkpoint_sha256']==a.expected_parent_sha
    assert ev['frames']==23200 and len(ev['streams'])==320 and ev['fp_calls']==0
    samples,_=load_fixed_manifest(a.manifest,a.expected_manifest_sha);assert len(samples)==64000
    audit=check_data_gate(a.index_root);assert all(ev[k]==audit[k] for k in ('split_hash','mesh_hash'))
    sampling_receipt=None
    if a.manifest_receipt is not None:
        sampling_receipt=json.loads(a.manifest_receipt.read_text())
        assert sampling_receipt['completed'] and sampling_receipt['entries']==64000
        assert sampling_receipt['seed']==a.seed and sampling_receipt['manifest_sha256']==a.expected_manifest_sha
        assert all(sampling_receipt[k]==audit[k] for k in ('split_hash','mesh_hash'))
    a.out.mkdir(parents=True,exist_ok=False);manifest=a.out/'training_samples.json';manifest.write_bytes(a.manifest.read_bytes())
    c=dict(parent['config'],seed=a.seed,architecture_id='stream_rk_pose_reference',world_size=2,batch_sequences_per_gpu=32,grad_accum_steps=1,
        burn_in_frames=8,supervised_unroll_frames=48,effective_sequences_per_step=64,nominal_supervised_updates_per_step=3072,
        max_stage_steps=1000,save_every=250,warmup_steps=100,fixed_sampling_manifest_sha256=a.expected_manifest_sha)
    assert c['temporal_occlusion_probability']==.5
    e=dict(kind='startup_supervision_x_startup_occlusion',parent=str(parent_path.resolve()),parent_sha256=a.expected_parent_sha,
        source_sha256=source_hash(),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],data_root=str(a.data_root.resolve()),
        index_root=str(a.index_root.resolve()),initial_poses=old['initial_poses'],initial_poses_sha256=old['initial_poses_sha256'],
        reference_evaluation=str(reference.resolve()),seed=a.seed,steps=1000,training_manifest=str(manifest.resolve()),
        training_manifest_sha256=a.expected_manifest_sha,arms={},foundationpose_calls=0,quality_head_enabled=False,test_access=False,
        protocol='Four identical pose-reference actors freshly initialized from the fixed spatial M1A1, with zero feedback outputs. Same 64000 fragments, seed/noise, 56 observations and 48 supervised targets per fragment. No current experiment weight selection.',
        factors=dict(S='Off: targets at observations 9-56. On: targets at observations 1-8 plus 40 later positions; omit observations 14,20,26,32,38,44,50,56. Keep 48 labels, same 56 observations and KV detach at boundary 8.',
            O='Off: original delayed synthetic timing. On: independently shift half of already selected occluders to observation index 0, preserving appearance, duration and velocity. Total augmentation probability .5, expected startup fraction .25.'),
        compute_limit='Matched observation and label counts; S1 also backpropagates the initial eight frames, so backward compute and memory are measured rather than claimed identical.')
    e['randomness_contract']=dict(model_initialization_seed=a.seed,training_rank_seeds=[a.seed,a.seed+1],
        dataloader_generator_seed=a.seed,sampling_receipt=sampling_receipt,
        augmentation='Stored per-fragment child seeds control pose noise and synthetic occlusion; these are not unique sampler IDs.',
        scope='Independent adaptation-stage seeds with one shared frozen M1A1 ancestor, not independently retrained ancestors.')
    if sampling_receipt is not None:
        (a.out/'sampling_receipt.json').write_bytes(a.manifest_receipt.read_bytes())
        e['sampling_receipt_sha256']=sha(a.out/'sampling_receipt.json')
    source_state=None
    for arm in ARMS:
        folder=a.out/arm;folder.mkdir();cfg=dict(c,preflight_receipt=str((folder/'approval.json').resolve()),
            startup_supervision_frames=8 if arm[1]=='1' else 0,temporal_occlusion_startup_probability=.5 if arm[3]=='1' else 0.)
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_stream_config(path)
        torch.manual_seed(a.seed);model=make_model(cfg);loaded=model.load_state_dict(parent['model'],strict=False)
        assert not loaded.unexpected_keys and all(k.startswith('pose_reference_feedback.') for k in loaded.missing_keys)
        assert all(torch.equal(t,model.state_dict()[name]) for name,t in parent['model'].items())
        if source_state is None:source_state={k:t.clone() for k,t in model.state_dict().items()}
        else:assert all(torch.equal(t,model.state_dict()[name]) for name,t in source_state.items())
        migration=dict(parent=dict(path=str(parent_path.resolve()),sha256=a.expected_parent_sha,architecture_id=parent['architecture_id'],
            new_stage_step=parent['new_stage_step'],split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash']),
            parameters={k:dict(status='loaded' if k in parent['model'] else 'initialized') for k in model.state_dict()},
            optimizer_restored=False,new_stage_step=0,all_parent_tensors_bitwise_equal=True)
        model.migration_status=migration['parameters'];save_init(folder/'init.pt',model,cfg,migration)
        e['arms'][arm]=dict(config=str(path.resolve()),init=str((folder/'init.pt').resolve()),init_sha256=sha(folder/'init.pt'),
            trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad},
            supervised_observation_indices=[i for i,flag in enumerate(supervision_positions(8,48,cfg['startup_supervision_frames'])) if flag])
    assert all(e['arms'][arm]['trainable']==e['arms'][ARMS[0]]['trainable'] for arm in ARMS)
    (a.out/'experiment.json').write_text(json.dumps(e,indent=2))
    (a.out/'evaluation_request.json').write_text(json.dumps(dict(initial_poses=e['initial_poses'],initial_poses_sha256=e['initial_poses_sha256'],
        scope='Reuse the already selected controlled-val initializer protocol after final fixed-step training.'),indent=2))
    print(json.dumps({k:v for k,v in e.items() if k!='arms'},indent=2))


if __name__=='__main__':main()
