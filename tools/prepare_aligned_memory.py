"""Paired continuation versus an added local CAD-alignment residual."""
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
    for key in ('parent-experiment','data-root','index-root','out'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);old=json.loads((a.parent_experiment/'experiment.json').read_text());audit=check_data_gate(a.index_root)
    assert json.loads((a.parent_experiment/'status.json').read_text())['phase']=='completed'
    parent_path=a.parent_experiment/'spatial/train/last.pt';expected='3371614e5efa788bf8c64cd10b64498040edf68a49d246c136d779e096ee0f2c';assert sha(parent_path)==expected
    parent=torch.load(parent_path,map_location='cpu',weights_only=False);assert parent['architecture_id']=='stream_rk_spatial' and parent['new_stage_step']==1000
    c=parent['config'].copy();c.update(world_size=4,batch_sequences_per_gpu=16,grad_accum_steps=1,effective_sequences_per_step=64,
        supervised_unroll_frames=48,nominal_supervised_updates_per_step=3072,max_stage_steps=1000,save_every=250,warmup_steps=100)
    assert c['temporal_occlusion_probability']==.5
    e=dict(parent=str(parent_path.resolve()),parent_sha256=expected,source_sha256=source_hash(),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],
        data_root=str(a.data_root.resolve()),index_root=str(a.index_root.resolve()),initial_poses=old['initial_poses'],initial_poses_sha256=old['initial_poses_sha256'],
        seed=42,steps=1000,arms={},reader_arm='aligned',temporal_occlusion_probability=.5,
        protocol='Same trained spatial+occlusion parent, 64 fragments/step, 8+48 frames, matched old-parameter LR, final step 1000; no intermediate val selection',
        trained_modules='Existing parent small modules in both arms; only aligned_readout parameters are new. RGB/geometry/spatial-fusion/temporal/pose-head core remains frozen.',
        quality_head_enabled=False,foundationpose_calls=0,test_access=False,reference_evaluation=str((a.parent_experiment/'spatial/s0_val').resolve()),
        sample_counter_range=[64000,127999],sampling_note='Advance sample counter with seed 42. Merely changing seed to 43 would shift the previous manifest by one item.')
    for name in ('control','aligned'):
        folder=a.out/name;folder.mkdir();cfg=dict(c,preflight_receipt=str((folder/'approval.json').resolve()))
        if name=='aligned':cfg.update(architecture_id='stream_rk_aligned',aligned_query_side=14,aligned_sigma=.05)
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_stream_config(path)
        torch.manual_seed(42);model=make_model(cfg);loaded=model.load_state_dict(parent['model'],strict=False)
        assert not loaded.unexpected_keys and all(k.startswith('aligned_readout.') for k in loaded.missing_keys)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in parent['model'].items())
        migration=dict(parent=dict(path=str(parent_path.resolve()),sha256=expected,architecture_id=parent['architecture_id'],new_stage_step=1000,
            split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash']),parameters={k:dict(status='loaded' if k in parent['model'] else 'initialized') for k in model.state_dict()},
            optimizer_restored=False,new_stage_step=0,all_parent_tensors_bitwise_equal=True)
        model.migration_status=migration['parameters'];save_init(folder/'init.pt',model,cfg,migration)
        e['arms'][name]=dict(config=str(path.resolve()),init=str((folder/'init.pt').resolve()),init_sha256=sha(folder/'init.pt'),trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad})
    ds=StreamClips(a.data_root,a.index_root,8,48,seed=42,start_sample=64000);items=[ds.choose(i) for i in range(64000)]
    assert set(ds.groups)=={s['object_id'] for s in ds.streams}
    e['eligible_training_objects']=sorted(ds.groups);e['eligible_streams']=sum(bool(len(v)) for v in ds.starts);e['total_training_streams']=len(ds.streams)
    old_items=json.loads(Path(old['training_manifest']).read_text());key=lambda x:(x['stream'],x['start'],x['seed'])
    e['exact_sample_noise_overlap_with_previous_stage']=len(set(map(key,items))&set(map(key,old_items)))
    assert e['exact_sample_noise_overlap_with_previous_stage']==0
    manifest=a.out/'training_samples.json';manifest.write_text(json.dumps(items));e['training_manifest']=str(manifest.resolve());e['training_manifest_sha256']=sha(manifest)
    (a.out/'experiment.json').write_text(json.dumps(e,indent=2));print(json.dumps({k:v for k,v in e.items() if k!='arms'},indent=2))


if __name__=='__main__':main()
