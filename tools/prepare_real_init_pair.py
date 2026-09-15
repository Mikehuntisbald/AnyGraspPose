"""Matched fixed-budget adaptation from the retained residual checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash,save_init
from lip.engine.stream_config import make_model,load_stream_config
from lip.engine.config import check_data_gate
from lip.data.external_initializers import initializer_key,request_real_initializer,load_train_initializers
from lip.data.stream_clips import StreamClips


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('parent','train-initializers','requests','data-root','index-root','val-initializers','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-parent-sha',required=True);a=p.parse_args();torch.set_num_threads(2)
    assert sha(a.parent)==a.expected_parent_sha;parent=torch.load(a.parent,map_location='cpu',weights_only=False)
    assert parent['architecture_id']=='stream_dual_cross_residual';audit=check_data_gate(a.index_root)
    assert all(parent[k]==audit[k] for k in ('split_hash','mesh_hash'))
    req=json.loads(a.requests.read_text());assert req['completed'] and req['split']=='train' and req['draws']==64000 and req['real_probability']==.5
    samples=json.loads(Path(req['training_manifest']).read_text());assert sha(req['training_manifest'])==req['training_manifest_sha256'] and len(samples)==64000
    data=StreamClips(a.data_root,a.index_root,8,48,fixed=samples,decode_threads=0)
    external=load_train_initializers(a.train_initializers,sha(a.train_initializers),data.streams,audit)
    assert external['requests_sha256']==sha(a.requests);requested=used=0;probe_indices=[]
    for i,item in enumerate(samples):
        if request_real_initializer(item['seed'],.5):
            requested+=1;s=data.streams[item['stream']];frame=int(data.poses[item['stream']]['frames'][item['start']]);key=initializer_key(s['stream_id'],frame)
            assert key in external['initializers']
            if external['initializers'][key] is not None:
                used+=1
                if len(probe_indices)<8:probe_indices.append(i)
    assert requested==req['requested_draws'] and used>0 and len(probe_indices)==8
    a.out.mkdir(parents=True,exist_ok=False);manifest=a.out/'training_samples.json';manifest.write_text(json.dumps(samples))
    c=dict(parent['config'],world_size=4,batch_sequences_per_gpu=16,grad_accum_steps=1,effective_sequences_per_step=64,
        burn_in_frames=8,supervised_unroll_frames=48,startup_supervision_frames=8,nominal_supervised_updates_per_step=3072,
        max_stage_steps=1000,save_every=250,warmup_steps=100,seed=42,freeze_rgb_steps=1000,freeze_rgb_for_stage=True,freeze_parent_steps=0,
        lr_loaded_modules=2e-6,lr_rgb_backbone=0.,lr_new_modules=1e-5,initial_pose_noise=True,augmentation=False,
        temporal_occlusion_probability=.5,temporal_occlusion_startup_probability=.5,prime_initial_observation=True,batch_current_features=True,
        fixed_sampling_manifest_sha256=sha(manifest),external_initializers=str(a.train_initializers.resolve()),external_initializers_sha256=sha(a.train_initializers))
    e=dict(source_sha256=source_hash(),parent=str(a.parent.resolve()),parent_sha256=sha(a.parent),split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],
        data_root=str(a.data_root.resolve()),index_root=str(a.index_root.resolve()),training_manifest=str(manifest.resolve()),training_manifest_sha256=sha(manifest),
        train_initializers=str(a.train_initializers.resolve()),train_initializers_sha256=sha(a.train_initializers),
        val_initializers=str(a.val_initializers.resolve()),val_initializers_sha256=sha(a.val_initializers),steps=1000,seed=42,arms={},
        requested_real_draws=requested,used_real_draws=used,missing_real_draws=requested-used,probe_indices=probe_indices,
        protocol='Both arms: same residual parent, same 64000 fragments, same synthetic occluders, fresh optimizer, fixed final1000, RGB frozen; non-RGB modules adapt at small LR. One retained initial observation plus 56 updates, exactly 48 supervised targets including the first eight. Actor pose feedback detaches.',
        intervention='control: noisy-GT initial poses; real_mix: deterministic 50% requests for actual train PoseCNN poses, missing detections explicitly fall back to the same noisy prior. No real-initializer noise is added. No FP use in either training arm.',
        evaluation='Full real-initialized val, same frozen first-legal PoseCNN initializers as pure FP and residual. Report all/occlusion/bad-start accuracy, rotation, center, recovery and isolated runtime. No automatic test launch.',
        promotion='Protect residual overall ADD and strict ADD-S point scores plus severe strict accuracy; require improved paired evidence on an FP deficit (occluded ADD or bad-start rotation/strict accuracy). All-metric superiority is claimed only when every predeclared FP scorecard entry is favorable; partial wins remain partial.')
    for name,probability in [('control',0.),('real_mix',.5)]:
        folder=a.out/name;folder.mkdir();cfg=dict(c,real_initialization_probability=probability,preflight_receipt=str((folder/'approval.json').resolve()))
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_stream_config(path)
        torch.manual_seed(42);model=make_model(cfg);model.load_state_dict(parent['model'],strict=True);model.rgb.requires_grad_(False)
        assert all(torch.equal(t,model.state_dict()[k]) for k,t in parent['model'].items())
        migration=dict(parent=dict(path=str(a.parent.resolve()),sha256=sha(a.parent),architecture_id=parent['architecture_id'],
            new_stage_step=parent['new_stage_step'],split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash']),
            parameters=parent.get('migration_status',{k:dict(status='loaded') for k in parent['model']}),optimizer_restored=False,
            all_parent_tensors_bitwise_equal=True,new_stage_step=0)
        model.migration_status=migration['parameters'];save_init(folder/'init.pt',model,cfg,migration)
        e['arms'][name]=dict(config=str(path.resolve()),init=str((folder/'init.pt').resolve()),init_sha256=sha(folder/'init.pt'),
            trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad})
    assert e['arms']['control']['trainable']==e['arms']['real_mix']['trainable']
    (a.out/'experiment.json').write_text(json.dumps(e,indent=2));print(json.dumps({k:v for k,v in e.items() if k!='arms'},indent=2))


if __name__=='__main__':main()
