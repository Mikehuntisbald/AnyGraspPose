"""Freeze the spatial-supervision x direct-parent-latent factorial before training."""
import json,sys
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash,save_init
from lip.engine.stream_config import make_model,load_stream_config


def main():
    root=Path(__file__).resolve().parents[1];out=root/'runs/factorial';out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    old=Path('/mnt/why/dexycb_lip/intraframe_alignment_20260915/runs/alignment_pair/experiment.json')
    e=json.loads(old.read_text());zero=torch.load(e['arms']['alignment']['init'],map_location='cpu',weights_only=False)
    parent=torch.load(e['parent'],map_location='cpu',weights_only=False)
    assert sha(e['parent'])==e['parent_sha256']=='89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868'
    info=Path(e['data_root'])/'bop/models/models_info.json';metadata=json.loads(info.read_text())
    symmetric=sorted(int(k) for k,v in metadata.items() if any(v.get(n) for n in ('symmetries_discrete','symmetries_continuous')))
    c=load_stream_config(e['arms']['alignment']['config'])
    c.update(world_size=2,grad_accum_steps=2,batch_sequences_per_gpu=16,effective_sequences_per_step=64,
        alignment_symmetric_object_ids=symmetric,alignment_models_info_sha256=sha(info))
    e=dict(e,arms={},source_sha256=source_hash(),old_experiment=str(old),old_experiment_sha256=sha(old),
        initial_branch_checkpoint=e['arms']['alignment']['init'],initial_branch_sha256=sha(e['arms']['alignment']['init']),
        models_info=str(info),models_info_sha256=sha(info),symmetric_object_ids=symmetric,
        protocol='2x2 seed42: auxiliary spatial attention supervision 0/.05 x direct parent latent on/off. Same zero-branch checkpoint, retained residual, all 64000 fixed train draws, effective batch64, final1000, S1O1, .5 real PoseCNN request, fresh Adam and frozen RGB. Each arm two H20 with two microbatches. No val-driven checkpoint selection.',
        auxiliary='Mean-head actual cross-attention distribution supervised by base-visible CAD representative points projected with training GT; bilinear target splat to observation grid; GT self-depth and augmented sensor-depth consistency; no object masks or hands. Symmetric objects from bound BOP metadata excluded from auxiliary only, remain in pose training and all evaluation.',
        evaluation='Full real non-GT initialization native s0 val, 320 streams/23200 frames per arm, zero FP, one correction/frame. Paired conditional supervision and latent effects plus interaction. Separate original parent/control references. No new official test.',
        limits='No direct parent latent does not remove geometry already in fused observation tokens. Auxiliary visibility can shrink with bad priors; coverage must be reported. One seed and one fixed loss weight; no sweep.')
    for name,spatial,latent in [('s0_l1',False,True),('s0_l0',False,False),('s1_l1',True,True),('s1_l0',True,False)]:
        folder=out/name;folder.mkdir();cfg=dict(c,alignment_spatial_weight=.05 if spatial else 0.,alignment_use_parent_latent=latent,
            preflight_receipt=str(folder/'approval.json'))
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));load_stream_config(path)
        torch.manual_seed(42);model=make_model(cfg);model.load_state_dict(zero['model'],strict=True)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in parent['model'].items())
        assert not model.rotation_alignment.output[-1].weight.count_nonzero()
        migration=dict(parent=zero['parent'],parameters=zero['migration_status'],all_parent_tensors_bitwise_equal=True,optimizer_restored=False)
        model.migration_status=zero['migration_status'];save_init(folder/'init.pt',model,cfg,migration)
        e['arms'][name]=dict(config=str(path),init=str(folder/'init.pt'),init_sha256=sha(folder/'init.pt'),spatial=spatial,parent_latent=latent)
    (out/'experiment.json').write_text(json.dumps(e,indent=2));print(json.dumps(dict(source=e['source_sha256'],symmetry=symmetric,arms=list(e['arms']))))


if __name__=='__main__':main()
