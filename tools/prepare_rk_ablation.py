"""Create four matched stages and immutable training samples from residual 1000."""
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.config import check_data_gate
from lip.engine.stream_config import make_model,load_stream_config
from lip.engine.stream_checkpoint import sha,source_hash,save_init
from lip.data.stream_clips import StreamClips

ARMS=('R0K0','R1K0','R0K1','R1K1')
PARENT_SHA='89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868'


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('parent','data-root','index-root','out'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    audit=check_data_gate(a.index_root)
    if sha(a.parent)!=PARENT_SHA:raise ValueError('Expected the frozen, selected residual 1000 parent')
    parent=torch.load(a.parent,map_location='cpu',weights_only=False)
    assert parent['architecture_id']=='stream_dual_cross_residual' and parent['new_stage_step']==1000
    assert all(parent[k]==audit[k] for k in ('split_hash','mesh_hash'))
    base=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/stream_lip_v2_residual_8800.yaml').read_text())
    base.update(architecture_id='stream_rk_factorial',world_size=2,batch_sequences_per_gpu=32,grad_accum_steps=1,
        effective_sequences_per_step=64,supervised_unroll_frames=32,nominal_supervised_updates_per_step=2048,
        max_stage_steps=1000,save_every=250,warmup_steps=100,freeze_parent_steps=0,freeze_rgb_steps=0,
        keyframe_slots=4,keyframe_min_gap=4,keyframe_max_age=64,support_tolerance=.05,lr_quality=.001)
    receipt=dict(parent_sha256=PARENT_SHA,source_sha256=source_hash(),data_root=str(a.data_root.resolve()),
        index_root=str(a.index_root.resolve()),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],
        seed=42,steps=1000,arms={},training_split='train',test_access=False,foundationpose_calls=0,
        budget='64 sequences and 2048 supervised frames per step; 8 burn-in + 32 supervised; 2 GPUs per arm',
        selection='Final step 1000 fixed in advance; no test-based selection; one training seed')
    for arm in ARMS:
        folder=a.out/arm;folder.mkdir()
        c=dict(base,observation_reliability=arm[1]=='1',keyframe_memory=arm[3]=='1',preflight_receipt=str((folder/'approval.json').resolve()))
        cfg=folder/'config.yaml';cfg.write_text(yaml.safe_dump(c,sort_keys=False));load_stream_config(cfg)
        torch.manual_seed(42);model=make_model(c)
        loaded=model.load_state_dict(parent['model'],strict=False)
        if loaded.unexpected_keys:raise ValueError(loaded.unexpected_keys)
        allowed=('anchor_attn.','anchor_gate.','anchor_time.','reliability_strength','update_strength')
        if any(not key.startswith(allowed) for key in loaded.missing_keys):raise ValueError(loaded.missing_keys)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in parent['model'].items())
        report={k:dict(status='loaded' if k in parent['model'] else 'initialized',shape=list(v.shape)) for k,v in model.state_dict().items()}
        model.migration_status=report
        migration=dict(parent=dict(path=str(a.parent.resolve()),sha256=PARENT_SHA,architecture_id=parent['architecture_id'],
            new_stage_step=1000,split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash']),parameters=report,
            optimizer_restored=False,new_stage_step=0,all_parent_tensors_bitwise_equal=True)
        init=folder/'init.pt';save_init(init,model,c,migration)
        trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
        receipt['arms'][arm]=dict(config=str(cfg.resolve()),init=str(init.resolve()),init_sha256=sha(init),
            trainable=trainable,trainable_numel=sum(trainable.values()))
    dataset=StreamClips(a.data_root,a.index_root,8,32,seed=42)
    # Includes repeats exactly as normal sampling does. Every arm consumes the
    # identical global sample indices, including each fixed noise seed.
    manifest=a.out/'training_samples.json'
    manifest.write_text(json.dumps([dataset.choose(i) for i in range(1000*64)]))
    receipt['training_manifest']=str(manifest.resolve());receipt['training_manifest_sha256']=sha(manifest)
    (a.out/'experiment.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
